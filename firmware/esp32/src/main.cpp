// One ESP32 does both jobs:
//   - core 0: plays the stepped sine sweep on the DAC (GPIO25 -> amp -> transducer)
//   - core 1: samples the four piezos at a fixed rate and streams the raw counts
// All processing (amplitude, phase, features) happens on the laptop (pc/).
//
// Channels: up_right GPIO34, up_left GPIO35, down_left GPIO32, down_right GPIO33
//
// Sweep: lead_ms of silence, then for each frequency f_start, f_start+f_step, ... <= f_stop:
//   tone_ms of sine (raised-cosine ramp of ramp_ms at both ends, phase 0 at the tone start),
//   then gap_ms of silence. Defaults match pc/piezo/schedule.py; the laptop sends cfg anyway.
//
// Serial 921600 baud. Text commands, end with newline:
//   i            INFO line (rate, pins, ADC read spacing, sweep settings)       [stopped only]
//   r2000        sample rate in Hz, 500..4000                                   [stopped only]
//   cfg <f_start> <f_stop> <f_step> <tone_ms> <gap_ms> <lead_ms> <amp> <ramp_ms>  [stopped only]
//   b            begin streaming
//   sweep        run one sweep (normally while streaming)
//   tone <hz>    continuous tone, for setup checks
//   off          stop any tone or sweep
//   x            stop streaming
// While streaming the port carries only binary blocks, so replies come as event blocks.
//
// Binary blocks (little endian):
//  Data:  A5 5A | uint16 n | uint32 seq | uint16 overruns | n x (uint32 t_us, uint16 ur,ul,dl,dr) | uint16 sum
//  Event: A5 5B | uint8 type | uint32 t_us | float freq_hz | uint16 sum
//         type 1 = sweep start (lead silence begins), 2 = tone start (phase 0 at t_us),
//              3 = tone end, 4 = sweep end, 5 = sweep aborted
//  sum = 16-bit sum of every byte after the two magic bytes, before the sum.
//  t_us is micros() on this board, the same clock for samples and tone, so the laptop
//  knows the drive phase at every sample. Channel k is read about k * adc_read_us after t_us.

#include <Arduino.h>
#include <math.h>

namespace {
constexpr uint8_t kPins[4] = {34, 35, 32, 33};  // ur, ul, dl, dr
constexpr uint8_t kDacPin = 25;
constexpr uint32_t kBaud = 921600;
constexpr const char *kFirmware = "piezo-sweep-2.0";

// ---------------------------------------------------------------- sweep settings
struct Config {
  float fStart = 20, fStop = 300, fStep = 10;
  uint32_t toneMs = 600, gapMs = 100, leadMs = 1000;
  uint8_t amp = 100;  // DAC counts around mid-scale 128
  uint32_t rampMs = 10;
} cfg;

enum class Mode : uint8_t { Idle, Sweep, Tone };
volatile Mode requested = Mode::Idle;
volatile bool stopRequested = false;
volatile float toneHz = 0;

enum Event : uint8_t { kSweepStart = 1, kToneStart = 2, kToneEnd = 3, kSweepEnd = 4, kSweepAborted = 5 };
struct EventMsg { uint8_t type; uint32_t tUs; float freq; };
QueueHandle_t eventQueue;

constexpr uint32_t kTable = 1024;
float sineTable[kTable];

// ---------------------------------------------------------------- drive (core 0)
void postEvent(uint8_t type, uint32_t tUs, float f) {
  EventMsg m{type, tUs, f};
  xQueueSend(eventQueue, &m, 0);
}

// Plays f for durMs (0 = until stopped). Returns false if stopped early.
bool playTone(float f, uint32_t durMs, uint32_t rampMs) {
  const uint32_t t0 = micros();
  postEvent(kToneStart, t0, f);
  const uint32_t durUs = durMs * 1000, rampUs = rampMs * 1000;
  bool completed = true;
  while (true) {
    const uint32_t dt = micros() - t0;
    if (durMs && dt >= durUs) break;
    if (stopRequested) { completed = false; break; }
    float env = 1.0f;
    if (durMs && rampUs) {
      const uint32_t edge = min(dt, durUs - dt);
      if (edge < rampUs) env = 0.5f - 0.5f * cosf(PI * edge / rampUs);
    }
    const float cycles = f * (dt * 1e-6f);
    const uint32_t idx = uint32_t((cycles - floorf(cycles)) * kTable) % kTable;
    dacWrite(kDacPin, uint8_t(128 + int(cfg.amp * env * sineTable[idx])));
  }
  dacWrite(kDacPin, 128);
  postEvent(kToneEnd, micros(), f);
  return completed;
}

bool silenceFor(uint32_t ms) {
  const uint32_t t0 = micros();
  while (micros() - t0 < ms * 1000) {
    if (stopRequested) return false;
  }
  return true;
}

void driveTask(void *) {
  for (;;) {
    const Mode m = requested;
    if (m == Mode::Idle) {
      vTaskDelay(1);
      continue;
    }
    stopRequested = false;
    if (m == Mode::Tone) {
      playTone(toneHz, 0, 0);
    } else {
      bool ok = true;
      postEvent(kSweepStart, micros(), 0);
      ok = silenceFor(cfg.leadMs);
      for (int k = 0; ok; ++k) {
        const float f = cfg.fStart + k * cfg.fStep;
        if (f > cfg.fStop + 1e-3f) break;
        ok = playTone(f, cfg.toneMs, cfg.rampMs) && silenceFor(cfg.gapMs);
      }
      postEvent(ok ? kSweepEnd : kSweepAborted, micros(), 0);
    }
    dacWrite(kDacPin, 128);
    requested = Mode::Idle;
    stopRequested = false;
  }
}

// ---------------------------------------------------------------- sampling (core 1)
constexpr uint16_t kFramesPerBlock = 50;
constexpr size_t kFrameBytes = 4 + 4 * 2;
constexpr size_t kHeaderBytes = 2 + 2 + 4 + 2;
uint8_t block[kHeaderBytes + kFramesPerBlock * kFrameBytes + 2];

uint32_t sampleRateHz = 2000;
uint32_t periodUs = 500;
bool streaming = false;
uint32_t nextSampleUs = 0;
uint32_t blockSeq = 0;
uint16_t overruns = 0;
uint16_t framesInBlock = 0;
float adcReadUs = 0;
String command;

inline void put16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
inline void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = (v >> (8 * i)) & 0xFF;
}

void writeWithSum(uint8_t *buf, size_t len) {  // len excludes the 2-byte sum
  uint16_t sum = 0;
  for (size_t i = 2; i < len; ++i) sum += buf[i];
  put16(buf + len, sum);
  Serial.write(buf, len + 2);
}

void sendBlock() {
  block[0] = 0xA5;
  block[1] = 0x5A;
  put16(block + 2, framesInBlock);
  put32(block + 4, blockSeq++);
  put16(block + 8, overruns);
  overruns = 0;
  writeWithSum(block, kHeaderBytes + framesInBlock * kFrameBytes);
  framesInBlock = 0;
}

void sendEvent(const EventMsg &m) {
  uint8_t buf[2 + 1 + 4 + 4 + 2];
  buf[0] = 0xA5;
  buf[1] = 0x5B;
  buf[2] = m.type;
  put32(buf + 3, m.tUs);
  memcpy(buf + 7, &m.freq, 4);
  writeWithSum(buf, 11);
}

void flushEvents() {
  EventMsg m;
  while (xQueueReceive(eventQueue, &m, 0) == pdTRUE) {
    if (streaming) sendEvent(m);
  }
}

void measureAdcSpacing() {
  const int reps = 200;
  const uint32_t t0 = micros();
  for (int r = 0; r < reps; ++r)
    for (uint8_t pin : kPins) analogRead(pin);
  adcReadUs = float(micros() - t0) / (reps * 4);
}

void printInfo() {
  Serial.printf("INFO,fw=%s,rate_hz=%lu,channels=ur;ul;dl;dr,pins=34;35;32;33,dac_pin=25,"
                "adc_read_us=%.2f,f_start=%.2f,f_stop=%.2f,f_step=%.2f,tone_ms=%lu,gap_ms=%lu,"
                "lead_ms=%lu,amp=%u,ramp_ms=%lu\n",
                kFirmware, (unsigned long)sampleRateHz, adcReadUs, cfg.fStart, cfg.fStop,
                cfg.fStep, (unsigned long)cfg.toneMs, (unsigned long)cfg.gapMs,
                (unsigned long)cfg.leadMs, cfg.amp, (unsigned long)cfg.rampMs);
}

void reply(const char *text) {
  if (!streaming) Serial.println(text);  // never mix text into the binary stream
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  if (cmd == "x") {
    if (streaming && framesInBlock > 0) sendBlock();
    flushEvents();
    streaming = false;
    Serial.flush();
    delay(20);
    Serial.println("\nSTOPPED");
  } else if (cmd == "b") {
    if (!streaming) {
      blockSeq = 0; overruns = 0; framesInBlock = 0;
      streaming = true;
      nextSampleUs = micros() + 1000;
    }
  } else if (cmd == "sweep") {
    if (requested == Mode::Idle) requested = Mode::Sweep;
    reply("OK,sweep");
  } else if (cmd == "off") {
    stopRequested = true;
    reply("OK,off");
  } else if (cmd.startsWith("tone")) {
    const float f = cmd.substring(4).toFloat();
    if (f >= 1 && f <= 2000 && requested == Mode::Idle) {
      toneHz = f;
      requested = Mode::Tone;
      reply("OK,tone (send off to stop)");
    } else {
      reply("ERR,tone needs 1..2000 Hz and no sweep running");
    }
  } else if (streaming) {
    return;  // settings only change while stopped
  } else if (cmd == "i") {
    printInfo();
  } else if (cmd.startsWith("r")) {
    sampleRateHz = constrain(cmd.substring(1).toInt(), 500L, 4000L);
    periodUs = 1000000 / sampleRateHz;
    sampleRateHz = 1000000 / periodUs;
    printInfo();
  } else if (cmd.startsWith("cfg")) {
    float a, b, c;
    unsigned long tone, gap, lead, amp, ramp;
    if (requested == Mode::Idle &&
        sscanf(cmd.c_str() + 3, "%f %f %f %lu %lu %lu %lu %lu", &a, &b, &c, &tone, &gap, &lead,
               &amp, &ramp) == 8 && c > 0 && b >= a && 2 * ramp <= tone) {
      cfg.fStart = a; cfg.fStop = b; cfg.fStep = c;
      cfg.toneMs = tone; cfg.gapMs = gap; cfg.leadMs = lead;
      cfg.amp = min(amp, 127UL); cfg.rampMs = ramp;
      printInfo();
    } else {
      Serial.println("ERR,cfg <f_start> <f_stop> <f_step> <tone_ms> <gap_ms> <lead_ms> <amp> <ramp_ms>");
    }
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
    } else if (command.length() < 96) {
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
  for (uint32_t i = 0; i < kTable; ++i) sineTable[i] = sinf(2 * PI * i / kTable);
  dacWrite(kDacPin, 128);
  eventQueue = xQueueCreate(32, sizeof(EventMsg));
  delay(200);
  measureAdcSpacing();
  // The tone loop never yields, so turn off the idle watchdog on core 0 (no WiFi is used).
  disableCore0WDT();
  xTaskCreatePinnedToCore(driveTask, "drive", 4096, nullptr, 1, nullptr, 0);
  Serial.println();
  printInfo();
}

void loop() {
  pollSerial();
  flushEvents();
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
