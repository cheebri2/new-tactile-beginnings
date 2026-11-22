// One ESP32 (system_board) does all the jobs:
//   - core 0: plays the stepped sine sweep on the DAC (GPIO26 -> PAM8403 -> transducer)
//   - core 1: samples the four piezos at a fixed rate and streams the raw counts, reads the
//     BNO055 IMU (angle + estimated x/y on the phantom, see tracker.h) and keeps a backup
//     log on the SPI flash (see flash_log.h)
// All piezo processing (amplitude, phase, features) happens on the laptop (pc/).
//
// Channels: up_right GPIO35, up_left GPIO32, down_left GPIO33, down_right GPIO34
// Amp: PAM8403 left input <- GPIO26 (DAC2) through 100 nF; SHDN <- GPIO4 (10k pull-up to VDC,
//   so the amp is on whenever the ESP32 is not driving GPIO4 low). In `amp auto` (default)
//   the amp is only switched on while a sweep or tone plays, so it adds no idle noise.
// IMU: BNO055 on UART2 (RX2 GPIO16 <- COM0, TX2 GPIO17 -> COM1), 100 Hz.
// Flash: W25Q32JVSS on VSPI (CS 5, CLK 18, MISO 19, MOSI 23).
// GPIO25 is V_Sense (battery / 2), reported in INFO as vbat.
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
//   test [amp]   continuous 150 Hz tone for hardware debugging; amp 0..127 (default cfg amp)
//   off          stop any tone or sweep
//   x            stop streaming
//   amp on|off|auto   PAM8403 SHDN control (auto = on only while playing)            [any time]
//   imu          one IMU line: angles, x/y, flags, calibration                       [stopped only]
//   imu zero     hold the head still at the phantom's lower-left corner: sets (0,0) and
//                zero angles at the next 0.25 s of stillness, replies OK,zero         [stopped only]
//   imu mode imu|ndof   fusion mode (imu = no magnetometer, default)                [stopped only]
//   mark <n>     write a mark record (e.g. trial number) into the flash log          [any time]
//   log on|off   flash logging of the IMU track (on by default once zeroed)          [any time]
//   flash info | flash erase | flash dump                                             [stopped only]
// While streaming the port carries only binary blocks, so replies come as event blocks.
//
// Binary blocks (little endian):
//  Data:  A5 5A | uint16 n | uint32 seq | uint16 overruns | n x (uint32 t_us, uint16 ur,ul,dl,dr) | uint16 sum
//  Event: A5 5B | uint8 type | uint32 t_us | float freq_hz | uint16 sum
//         type 1 = sweep start (lead silence begins), 2 = tone start (phase 0 at t_us),
//              3 = tone end, 4 = sweep end, 5 = sweep aborted
//  IMU:   A5 5C | uint32 t_us | int16 yaw, pitch, roll (0.01 deg) | float x_mm, y_mm
//              | int16 ax, ay (mm/s^2, phantom frame) | uint8 flags | uint8 calib | uint16 sum
//         about 100 per second. flags: 1 zeroed, 2 moving, 4 drive on (held still),
//         8 last move was long (> 5 s, less trustworthy), 16 gravity check odd at zero,
//         32 IMU data stale. Older laptop code skips these blocks.
//  sum = 16-bit sum of every byte after the two magic bytes, before the sum.
//  t_us is micros() on this board, the same clock for samples and tone, so the laptop
//  knows the drive phase at every sample. Channel k is read about k * adc_read_us after t_us.

#include <Arduino.h>
#include <math.h>

#include "bno055.h"
#include "flash_log.h"
#include "tracker.h"

namespace {
constexpr uint8_t kPins[4] = {35, 32, 33, 34};  // ur, ul, dl, dr
constexpr uint8_t kDacPin = 26;    // ACT_INPUT -> PAM8403 INL
constexpr uint8_t kShdnPin = 4;    // PAM8403 SHDN (active low)
constexpr uint8_t kVSensePin = 25; // battery through 100k/100k
constexpr int kImuRxPin = 16, kImuTxPin = 17;
constexpr uint32_t kBaud = 921600;
constexpr const char *kFirmware = "piezo-sweep-3.1";
constexpr float kTestHz = 150;

// ---------------------------------------------------------------- sweep settings
struct Config {
  float fStart = 50, fStop = 400, fStep = 14;
  uint32_t toneMs = 600, gapMs = 100, leadMs = 1000;
  uint8_t amp = 100;  // DAC counts around mid-scale 128
  uint32_t rampMs = 10;
} cfg;

enum class Mode : uint8_t { Idle, Sweep, Tone };
volatile Mode requested = Mode::Idle;
volatile bool stopRequested = false;
volatile float toneHz = 0;
volatile uint8_t toneAmp = 0;  // amplitude for continuous tones

enum Event : uint8_t { kSweepStart = 1, kToneStart = 2, kToneEnd = 3, kSweepEnd = 4, kSweepAborted = 5 };
struct EventMsg { uint8_t type; uint32_t tUs; float freq; };
QueueHandle_t eventQueue;

constexpr uint32_t kTable = 1024;
float sineTable[kTable];

// ---------------------------------------------------------------- amp shutdown pin
enum class AmpMode : uint8_t { Auto, On, Off };
volatile AmpMode ampMode = AmpMode::Auto;

volatile bool ampIsOn = false;  // digitalRead() of an output pin is not reliable on ESP32

void ampEnable(bool on) {
  digitalWrite(kShdnPin, on ? HIGH : LOW);
  ampIsOn = on;
}

// Applies the mode for "playing" or "idle". Returns true if the amp just woke up.
bool ampFor(bool playing) {
  const bool want = ampMode == AmpMode::On || (ampMode == AmpMode::Auto && playing);
  const bool wasOn = ampIsOn;
  ampEnable(want);
  return want && !wasOn;
}

// ---------------------------------------------------------------- drive (core 0)
void postEvent(uint8_t type, uint32_t tUs, float f) {
  EventMsg m{type, tUs, f};
  xQueueSend(eventQueue, &m, 0);
}

// Plays f for durMs (0 = until stopped). Returns false if stopped early.
bool playTone(float f, uint32_t durMs, uint32_t rampMs, uint8_t amp) {
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
    dacWrite(kDacPin, uint8_t(128 + int(amp * env * sineTable[idx])));
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
    // PAM8403 start-up is a few tens of ms; a sweep's lead silence (1 s) covers it.
    if (ampFor(true) && m == Mode::Tone) vTaskDelay(pdMS_TO_TICKS(60));
    if (m == Mode::Tone) {
      playTone(toneHz, 0, 0, toneAmp);
    } else {
      bool ok = true;
      postEvent(kSweepStart, micros(), 0);
      ok = silenceFor(cfg.leadMs);
      for (int k = 0; ok; ++k) {
        const float f = cfg.fStart + k * cfg.fStep;
        if (f > cfg.fStop + 1e-3f) break;
        ok = playTone(f, cfg.toneMs, cfg.rampMs, cfg.amp) && silenceFor(cfg.gapMs);
      }
      postEvent(ok ? kSweepEnd : kSweepAborted, micros(), 0);
    }
    dacWrite(kDacPin, 128);
    ampFor(false);
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

// ---------------------------------------------------------------- IMU + flash state
Bno055 bno(Serial2);
Tracker tracker;
FlashLog flashLog;
TrackerOut pose{};
ImuRaw lastImu{};
uint32_t lastImuUs = 0;
uint32_t imuCount = 0;
bool logEnabled = true;
bool zeroReplyPending = false;
uint32_t zeroRequestMs = 0;
uint32_t zeroImuCount = 0;
constexpr uint32_t kLogEvery = 4;  // flash log at 25 Hz (IMU at 100 Hz)

inline int16_t cdeg(float d) { return int16_t(constrain(lroundf(d * 100), -32767L, 32767L)); }

LogRec makeRec(uint8_t type, float value = 0) {
  LogRec r{};
  r.type = type;
  r.flags = pose.flags;
  r.calib = lastImu.calib;
  r.tMs = millis();
  r.yawCdeg = cdeg(pose.yawDeg);
  r.pitchCdeg = cdeg(pose.pitchDeg);
  r.rollCdeg = cdeg(pose.rollDeg);
  r.xMm = pose.xMm;
  r.yMm = pose.yMm;
  r.value = value;
  return r;
}

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
    if (logEnabled && m.type == kSweepStart) flashLog.append(makeRec(kLogSweepStart));
    if (logEnabled && (m.type == kSweepEnd || m.type == kSweepAborted)) {
      flashLog.append(makeRec(kLogSweepEnd, m.type == kSweepEnd ? 1 : 0));
      flashLog.flush();
    }
  }
}

void sendImu() {
  uint8_t buf[2 + 24 + 2];
  buf[0] = 0xA5;
  buf[1] = 0x5C;
  put32(buf + 2, lastImuUs);
  put16(buf + 6, uint16_t(cdeg(pose.yawDeg)));
  put16(buf + 8, uint16_t(cdeg(pose.pitchDeg)));
  put16(buf + 10, uint16_t(cdeg(pose.rollDeg)));
  memcpy(buf + 12, &pose.xMm, 4);
  memcpy(buf + 16, &pose.yMm, 4);
  put16(buf + 20, uint16_t(int16_t(constrain(lroundf(pose.axMs2 * 1000), -32767L, 32767L))));
  put16(buf + 22, uint16_t(int16_t(constrain(lroundf(pose.ayMs2 * 1000), -32767L, 32767L))));
  buf[24] = pose.flags;
  buf[25] = lastImu.calib;
  writeWithSum(buf, 26);
}

void printImu() {
  Serial.printf("IMU,ok=%d,mode=%s,yaw_deg=%.2f,pitch_deg=%.2f,roll_deg=%.2f,x_mm=%.1f,y_mm=%.1f,"
                "zeroed=%d,moving=%d,long_move=%d,gravity_odd=%d,stale=%d,calib_sys=%u,"
                "calib_gyr=%u,calib_acc=%u,calib_mag=%u,acc_dev=%.3f,gyro_dev=%.2f,samples=%lu,errors=%lu\n",
                bno.ok(), bno.mode() == Bno055::kModeNdof ? "ndof" : "imu", pose.yawDeg,
                pose.pitchDeg, pose.rollDeg, pose.xMm, pose.yMm, !!(pose.flags & Tracker::kZeroed),
                !!(pose.flags & Tracker::kMoving), !!(pose.flags & Tracker::kLongMove),
                !!(pose.flags & Tracker::kGravityOdd), !!(pose.flags & Tracker::kStale),
                lastImu.calib >> 6, (lastImu.calib >> 4) & 3, (lastImu.calib >> 2) & 3,
                lastImu.calib & 3, tracker.accDev(), tracker.gyroDev(), (unsigned long)imuCount,
                (unsigned long)bno.errors());
}

// Called every loop on core 1: never blocks.
void serviceImu() {
  ImuRaw r;
  if (bno.poll(r)) {
    lastImu = r;
    lastImuUs = r.tUs;
    pose = tracker.update(r.tUs, r.quat, r.liaMs2, r.grvMs2, r.gyroDps, requested != Mode::Idle);
    ++imuCount;
    if (streaming) sendImu();
    if (logEnabled && tracker.zeroed() && imuCount % kLogEvery == 0) flashLog.append(makeRec(kLogImu));
  }
  if (bno.ok() && micros() - lastImuUs > 100000) pose.flags |= Tracker::kStale;
  if (tracker.zeroDone()) {
    flashLog.append(makeRec(kLogZero));
    flashLog.flush();
    if (zeroReplyPending && !streaming) {
      Serial.printf("OK,zero,yaw_deg=%.2f,pitch_deg=%.2f,roll_deg=%.2f,gravity_odd=%d\n", pose.yawDeg,
                    pose.pitchDeg, pose.rollDeg, !!(pose.flags & Tracker::kGravityOdd));
    }
    zeroReplyPending = false;
  }
  if (zeroReplyPending && millis() - zeroRequestMs > 8000) {
    zeroReplyPending = false;
    if (streaming) return;
    const unsigned long n = imuCount - zeroImuCount;
    if (n == 0) {
      Serial.printf("ERR,zero: no IMU data arriving (BNO055 read errors=%lu)\n", (unsigned long)bno.errors());
    } else {
      Serial.printf("ERR,zero: not still for 0.25 s in 8 s (imu_samples=%lu, acc_dev=%.2f m/s2 "
                    "limit %.2f, gyro_dev=%.2f deg/s limit %.1f)\n",
                    n, tracker.accDev(), tracker.stillAccMs2, tracker.gyroDev(), tracker.stillGyroDps);
    }
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
  const char *amp = ampMode == AmpMode::Auto ? "auto" : (ampMode == AmpMode::On ? "on" : "off");
  Serial.printf("INFO,fw=%s,rate_hz=%lu,channels=ur;ul;dl;dr,pins=35;32;33;34,dac_pin=%u,"
                "shdn_pin=%u,amp_mode=%s,amp_on=%d,adc_read_us=%.2f,f_start=%.2f,f_stop=%.2f,f_step=%.2f,"
                "tone_ms=%lu,gap_ms=%lu,lead_ms=%lu,amp=%u,ramp_ms=%lu,imu=%s,imu_mode=%s,"
                "flash=%s,flash_used=%lu,vbat=%.2f\n",
                kFirmware, (unsigned long)sampleRateHz, kDacPin, kShdnPin, amp, int(ampIsOn), adcReadUs,
                cfg.fStart, cfg.fStop, cfg.fStep, (unsigned long)cfg.toneMs,
                (unsigned long)cfg.gapMs, (unsigned long)cfg.leadMs, cfg.amp,
                (unsigned long)cfg.rampMs, bno.ok() ? "bno055" : "missing",
                bno.mode() == Bno055::kModeNdof ? "ndof" : "imu", flashLog.ok() ? "w25q32" : "missing",
                (unsigned long)flashLog.used(), 2.0f * analogReadMilliVolts(kVSensePin) / 1000.0f);
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
  } else if (cmd.startsWith("amp ")) {
    const String m = cmd.substring(4);
    if (m == "on") ampMode = AmpMode::On;
    else if (m == "off") ampMode = AmpMode::Off;
    else if (m == "auto") ampMode = AmpMode::Auto;
    else { reply("ERR,amp on|off|auto"); return; }
    ampFor(requested != Mode::Idle);
    reply("OK,amp");
  } else if (cmd.startsWith("mark")) {
    flashLog.append(makeRec(kLogMark, cmd.substring(4).toFloat()));
    flashLog.flush();
    reply("OK,mark");
  } else if (cmd == "log on" || cmd == "log off") {
    logEnabled = cmd == "log on";
    if (!logEnabled) flashLog.flush();
    reply("OK,log");
  } else if (cmd == "sweep") {
    if (requested == Mode::Idle) requested = Mode::Sweep;
    reply("OK,sweep");
  } else if (cmd == "off") {
    stopRequested = true;
    reply("OK,off");
  } else if (cmd.startsWith("test")) {
    const String arg = cmd.substring(4);
    const long amp = arg.length() ? arg.toInt() : cfg.amp;
    if (amp >= 0 && amp <= 127 && requested == Mode::Idle) {
      toneHz = kTestHz;
      toneAmp = amp;
      requested = Mode::Tone;
      if (!streaming) Serial.printf("OK,test 150 Hz amp=%ld (send off to stop)\n", amp);
    } else {
      reply("ERR,test needs amp 0..127 and no sweep or tone running");
    }
  } else if (cmd.startsWith("tone")) {
    const float f = cmd.substring(4).toFloat();
    if (f >= 1 && f <= 2000 && requested == Mode::Idle) {
      toneHz = f;
      toneAmp = cfg.amp;
      requested = Mode::Tone;
      reply("OK,tone (send off to stop)");
    } else {
      reply("ERR,tone needs 1..2000 Hz and no sweep running");
    }
  } else if (streaming) {
    return;  // settings only change while stopped
  } else if (cmd == "i") {
    printInfo();
  } else if (cmd == "imu") {
    printImu();
  } else if (cmd == "imu zero") {
    if (!bno.ok()) {
      Serial.println("ERR,zero: no BNO055 (check UART2 wiring, PS0/PS1)");
    } else {
      tracker.requestZero();
      zeroReplyPending = true;
      zeroRequestMs = millis();
      zeroImuCount = imuCount;
    }
  } else if (cmd.startsWith("imu mode")) {
    const String m = cmd.substring(8);
    const bool ndof = m.indexOf("ndof") >= 0;
    if (bno.begin(kImuRxPin, kImuTxPin, ndof ? Bno055::kModeNdof : Bno055::kModeImu)) {
      tracker = Tracker();  // orientation reference changed: zero again
      Serial.printf("OK,imu mode %s (zero again)\n", ndof ? "ndof" : "imu");
    } else {
      Serial.println("ERR,BNO055 did not answer");
    }
  } else if (cmd == "flash info") {
    Serial.printf("FLASHINFO,ok=%d,jedec=%06lX,used=%lu,size=%lu,full=%d\n", flashLog.ok(),
                  (unsigned long)flashLog.jedec(), (unsigned long)flashLog.used(),
                  (unsigned long)FlashLog::kSize, flashLog.full());
  } else if (cmd == "flash erase") {
    Serial.print("erasing");
    const bool ok = flashLog.eraseAll(Serial);
    Serial.println(ok ? "\nOK,erased" : "\nERR,erase failed or no flash");
    if (ok) flashLog.append(makeRec(kLogBoot));
  } else if (cmd == "flash dump") {
    flashLog.flush();
    Serial.printf("FLASH,bytes=%lu\n", (unsigned long)flashLog.used());
    flashLog.dump(Serial);
    Serial.println("\nEND");
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
  pinMode(kShdnPin, OUTPUT);
  ampEnable(ampMode == AmpMode::On);
  analogSetPinAttenuation(kVSensePin, ADC_11db);
  eventQueue = xQueueCreate(32, sizeof(EventMsg));
  delay(200);
  measureAdcSpacing();
  bno.begin(kImuRxPin, kImuTxPin, Bno055::kModeImu);
  if (flashLog.begin()) flashLog.append(makeRec(kLogBoot));
  // The tone loop never yields, so turn off the idle watchdog on core 0 (no WiFi is used).
  disableCore0WDT();
  xTaskCreatePinnedToCore(driveTask, "drive", 4096, nullptr, 1, nullptr, 0);
  Serial.println();
  printInfo();
}

void loop() {
  pollSerial();
  flushEvents();
  serviceImu();
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
