// Drive ESP32: plays a stepped sine sweep on the DAC (GPIO25 -> amp -> transducer).
// It shares no wiring with the sensing ESP32; the laptop finds the sweep in the
// piezo data by its timing pattern, so no sync wire is needed.
//
// Sweep timing (must match pc/piezo/schedule.py, the laptop sends cfg first):
//   lead_ms of silence, then for each frequency f_start, f_start+f_step, ... <= f_stop:
//   tone_ms of sine (raised-cosine ramp of ramp_ms at both ends), then gap_ms of silence.
//   Every tone starts at phase 0.
//
// Serial 115200, commands end with newline:
//   cfg <f_start> <f_stop> <f_step> <tone_ms> <gap_ms> <lead_ms> <amp> <ramp_ms>
//   sweep        run one sweep (prints SWEEP_START ... SWEEP_END), x aborts
//   tone <hz>    continuous tone until off (for setup checks)
//   off          silence
//   info         print the current settings
// Pressing BOOT (GPIO0) also starts one sweep, for running without the laptop link.
//
// amp is the sine amplitude in DAC counts (0..127) around mid-scale 128.
// The DAC output sits at ~1.65 V DC; the amp input must be AC coupled
// (most amp boards are; otherwise add a 1-10 uF series capacitor).

#include <Arduino.h>
#include <math.h>

namespace {
constexpr uint8_t kDacPin = 25;
constexpr uint8_t kBootPin = 0;
constexpr uint32_t kTable = 1024;
constexpr const char *kFirmware = "drive-1.0";

struct Config {
  float fStart = 20, fStop = 300, fStep = 10;
  uint32_t toneMs = 600, gapMs = 100, leadMs = 1000;
  uint8_t amp = 100;
  uint32_t rampMs = 10;
} cfg;

float sineTable[kTable];
String command;
bool abortRequested = false;

void printInfo() {
  Serial.printf("INFO,fw=%s,f_start=%.2f,f_stop=%.2f,f_step=%.2f,tone_ms=%lu,gap_ms=%lu,"
                "lead_ms=%lu,amp=%u,ramp_ms=%lu\n",
                kFirmware, cfg.fStart, cfg.fStop, cfg.fStep, (unsigned long)cfg.toneMs,
                (unsigned long)cfg.gapMs, (unsigned long)cfg.leadMs, cfg.amp,
                (unsigned long)cfg.rampMs);
}

void silence() { dacWrite(kDacPin, 128); }

void pollAbort() {
  while (Serial.available()) {
    if (Serial.read() == 'x') abortRequested = true;
  }
}

// Play f Hz for durMs (0 = until a serial byte arrives). Phase 0 at the start.
void playTone(float f, uint32_t durMs, uint32_t rampMs) {
  const uint32_t t0 = micros();
  const uint32_t durUs = durMs * 1000, rampUs = rampMs * 1000;
  uint32_t lastPoll = t0;
  while (true) {
    const uint32_t dt = micros() - t0;
    if (durMs && dt >= durUs) break;
    float env = 1.0f;
    if (durMs && rampUs) {
      const uint32_t edge = min(dt, durUs - dt);
      if (edge < rampUs) env = 0.5f - 0.5f * cosf(PI * edge / rampUs);
    }
    const float cycles = f * (dt * 1e-6f);
    const uint32_t idx = uint32_t((cycles - floorf(cycles)) * kTable) % kTable;
    dacWrite(kDacPin, uint8_t(128 + int(cfg.amp * env * sineTable[idx])));
    if (dt - (lastPoll - t0) > 20000) {  // check serial every 20 ms
      lastPoll = micros();
      if (!durMs && Serial.available()) break;
      pollAbort();
      if (abortRequested) break;
    }
  }
  silence();
}

void waitMs(uint32_t ms) {
  const uint32_t t0 = millis();
  while (millis() - t0 < ms && !abortRequested) {
    pollAbort();
    delay(1);
  }
}

void runSweep() {
  abortRequested = false;
  silence();
  Serial.printf("SWEEP_START,%lu\n", (unsigned long)micros());
  waitMs(cfg.leadMs);
  int k = 0;
  for (float f = cfg.fStart; f <= cfg.fStop + 1e-3f && !abortRequested; f = cfg.fStart + (++k) * cfg.fStep) {
    playTone(f, cfg.toneMs, cfg.rampMs);
    waitMs(cfg.gapMs);
  }
  Serial.println(abortRequested ? "SWEEP_ABORTED" : "SWEEP_END");
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  if (cmd == "sweep") {
    runSweep();
  } else if (cmd == "off") {
    silence();
    Serial.println("OK,off");
  } else if (cmd == "info") {
    printInfo();
  } else if (cmd.startsWith("tone")) {
    const float f = cmd.substring(4).toFloat();
    if (f < 1 || f > 2000) {
      Serial.println("ERR,tone needs a frequency in Hz");
      return;
    }
    Serial.printf("OK,tone,%.2f (send any line to stop)\n", f);
    playTone(f, 0, 0);
    while (Serial.available()) Serial.read();
    Serial.println("OK,off");
  } else if (cmd.startsWith("cfg")) {
    float a, b, c;
    unsigned long tone, gap, lead, amp, ramp;
    if (sscanf(cmd.c_str() + 3, "%f %f %f %lu %lu %lu %lu %lu", &a, &b, &c, &tone, &gap,
               &lead, &amp, &ramp) == 8 && c > 0 && b >= a && 2 * ramp <= tone) {
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
}  // namespace

void setup() {
  Serial.begin(115200);
  pinMode(kBootPin, INPUT_PULLUP);
  for (uint32_t i = 0; i < kTable; ++i) sineTable[i] = sinf(2 * PI * i / kTable);
  silence();
  delay(100);
  printInfo();
}

void loop() {
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handleCommand(command);
      command = "";
    } else if (command.length() < 96) {
      command += c;
    }
  }
  if (digitalRead(kBootPin) == LOW) {
    delay(30);
    if (digitalRead(kBootPin) == LOW) {
      while (digitalRead(kBootPin) == LOW) delay(5);
      runSweep();
    }
  }
}
