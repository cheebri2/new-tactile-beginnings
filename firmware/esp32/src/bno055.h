// BNO055 over UART (board: PS1 = 3V3, PS0 = GND -> UART mode, 115200 8N1).
//   BNO055 COM0 (its TX) -> ESP32 RX2 = GPIO16
//   BNO055 COM1 (its RX) <- ESP32 TX2 = GPIO17
// RESET and BOOT_LOAD are pulled up with no GPIO, so resets go through SYS_TRIGGER.
// XIN32/XOUT32 have no crystal, so the internal oscillator is used.
//
// Protocol (BNO055 datasheet 4.7):
//   write: AA 00 reg len data...  -> EE 01 (01 = OK, anything else is an error)
//   read:  AA 01 reg len          -> BB len data...   or EE <status> on error
// Reads after init are non-blocking: poll() sends a burst request every period and
// parses the reply a few bytes at a time, so the 2 kHz piezo sampling keeps running.

#pragma once
#include <Arduino.h>

struct ImuRaw {
  uint32_t tUs;      // micros() when the registers were read
  float gyroDps[3];  // sensor frame, deg/s
  float quat[4];     // w, x, y, z (unit), sensor -> world
  float liaMs2[3];   // linear acceleration (gravity removed), sensor frame, m/s^2
  float grvMs2[3];   // gravity vector, sensor frame, m/s^2
  uint8_t calib;     // CALIB_STAT: sys<<6 | gyr<<4 | acc<<2 | mag
};

class Bno055 {
 public:
  static constexpr uint8_t kModeConfig = 0x00, kModeImu = 0x08, kModeNdof = 0x0C;

  explicit Bno055(HardwareSerial &port) : s_(port) {}

  // Blocking init. mode: kModeImu (accel+gyro, default: no magnetometer, so the
  // transducer and speaker magnets cannot pull the heading) or kModeNdof.
  bool begin(int rxPin, int txPin, uint8_t mode);
  bool ok() const { return ok_; }
  uint8_t mode() const { return mode_; }
  uint32_t errors() const { return errors_; }

  // Call often. Returns true once per completed read, with the sample in `out`.
  bool poll(ImuRaw &out, uint32_t periodUs = 10000);

 private:
  bool writeReg(uint8_t reg, uint8_t val);
  bool readRegs(uint8_t reg, uint8_t n, uint8_t *buf, uint32_t timeoutMs = 60);
  void decode(const uint8_t *b, ImuRaw &out) const;

  HardwareSerial &s_;
  bool ok_ = false;
  uint8_t mode_ = kModeImu;
  uint32_t errors_ = 0;

  // Burst read 0x14..0x35: gyro, euler (unused), quaternion, lia, gravity, temp, calib.
  static constexpr uint8_t kBurstReg = 0x14, kBurstLen = 0x35 - 0x14 + 1;  // 34 bytes
  enum class St : uint8_t { Idle, Head, Len, Data, Err } st_ = St::Idle;
  uint32_t lastReqUs_ = 0, reqUs_ = 0;
  uint8_t rx_[kBurstLen];
  uint8_t rxLen_ = 0, rxPos_ = 0;
};
