// Sensing ESP32: streams raw ADC samples from the four piezos to the laptop.
// All processing (alignment, lock-in, features) happens in pc/.
//
// Channel order everywhere: up_right, up_left, down_left, down_right
//                           GPIO34    GPIO35   GPIO32     GPIO33
//
// Serial: 921600 baud. Text commands (end with newline):
//   i         print one INFO line (rate, pins, ADC read spacing)
//   r2000     set sample rate in Hz (500..4000), only while stopped
//   b         begin streaming binary blocks
//   x         stop streaming
//
// Binary block (little endian), sent only while streaming:
//   uint8  0xA5, 0x5A        magic
//   uint16 n                 frames in this block
//   uint32 seq               block counter (gaps = dropped blocks)
//   uint16 overruns          samples that started late since the last block
//   n frames of:
//     uint32 t_us            micros() just before reading up_right
//     uint16 ur, ul, dl, dr  raw 12-bit ADC counts
//   uint16 checksum          sum of every byte from n up to the last frame
//
// Channel k is read about k * adc_read_us after t_us; the INFO line reports
// adc_read_us so the laptop can correct the small phase skew between channels.

#include <Arduino.h>

namespace {
constexpr uint8_t kPins[4] = {34, 35, 32, 33};  // ur, ul, dl, dr
constexpr uint32_t kBaud = 921600;
constexpr uint16_t kFramesPerBlock = 50;
constexpr size_t kFrameBytes = 4 + 4 * 2;
constexpr size_t kHeaderBytes = 2 + 2 + 4 + 2;
constexpr size_t kBlockBytes = kHeaderBytes + kFramesPerBlock * kFrameBytes + 2;
constexpr const char *kFirmware = "sense-1.0";

uint32_t sampleRateHz = 2000;
uint32_t periodUs = 1000000 / 2000;
bool streaming = false;
uint32_t nextSampleUs = 0;
uint32_t blockSeq = 0;
uint16_t overruns = 0;
uint16_t framesInBlock = 0;
uint8_t block[kBlockBytes];
float adcReadUs = 0;
String command;

inline void put16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
inline void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = (v >> (8 * i)) & 0xFF;
}

void measureAdcSpacing() {
  // Average time per analogRead, so the laptop knows when each channel was read.
  const int reps = 200;
  const uint32_t t0 = micros();
  for (int r = 0; r < reps; ++r)
    for (uint8_t pin : kPins) analogRead(pin);
  adcReadUs = float(micros() - t0) / (reps * 4);
}

void printInfo() {
  Serial.printf("INFO,fw=%s,rate_hz=%lu,channels=ur;ul;dl;dr,pins=34;35;32;33,"
                "adc_read_us=%.2f,frames_per_block=%u\n",
                kFirmware, (unsigned long)sampleRateHz, adcReadUs, kFramesPerBlock);
}

void sendBlock() {
  block[0] = 0xA5;
  block[1] = 0x5A;
  put16(block + 2, framesInBlock);
  put32(block + 4, blockSeq++);
  put16(block + 8, overruns);
  overruns = 0;
  const size_t end = kHeaderBytes + framesInBlock * kFrameBytes;
  uint16_t sum = 0;
  for (size_t i = 2; i < end; ++i) sum += block[i];
  put16(block + end, sum);
  Serial.write(block, end + 2);
  framesInBlock = 0;
}

void startStream() {
  blockSeq = 0;
  overruns = 0;
  framesInBlock = 0;
  streaming = true;
  nextSampleUs = micros() + 1000;
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  if (cmd == "x") {
    if (streaming && framesInBlock > 0) sendBlock();
    streaming = false;
    delay(20);
    Serial.println("\nSTOPPED");
  } else if (streaming) {
    return;  // only x is accepted while streaming, so text never mixes into the binary
  } else if (cmd == "i") {
    printInfo();
  } else if (cmd == "b") {
    startStream();
  } else if (cmd.startsWith("r")) {
    sampleRateHz = constrain(cmd.substring(1).toInt(), 500L, 4000L);
    periodUs = 1000000 / sampleRateHz;
    sampleRateHz = 1000000 / periodUs;  // report the rate actually used
    printInfo();
  } else {
    Serial.printf("ERR,unknown command %s\n", cmd.c_str());
  }
}

void pollSerial() {
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handleCommand(command);
      command = "";
    } else if (command.length() < 24) {
      command += c;
    }
  }
}
}  // namespace

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(kBaud);
  analogReadResolution(12);
  for (uint8_t pin : kPins) analogSetPinAttenuation(pin, ADC_11db);
  delay(200);
  measureAdcSpacing();
  Serial.println();
  printInfo();
}

void loop() {
  pollSerial();
  if (!streaming) return;

  // Pace samples on micros(). Signed difference handles the 71-minute wrap.
  if (int32_t(micros() - nextSampleUs) < 0) return;
  if (int32_t(micros() - nextSampleUs) > int32_t(periodUs)) ++overruns;
  nextSampleUs += periodUs;

  uint8_t *f = block + kHeaderBytes + framesInBlock * kFrameBytes;
  put32(f, micros());
  for (int k = 0; k < 4; ++k) put16(f + 4 + 2 * k, analogRead(kPins[k]));
  if (++framesInBlock == kFramesPerBlock) sendBlock();
}
