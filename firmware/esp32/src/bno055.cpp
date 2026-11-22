#include "bno055.h"

namespace {
constexpr uint8_t kRegChipId = 0x00, kRegPage = 0x07, kRegUnitSel = 0x3B, kRegOprMode = 0x3D,
                  kRegPwrMode = 0x3E, kRegSysTrigger = 0x3F;
constexpr uint8_t kChipId = 0xA0;

inline int16_t le16(const uint8_t *p) { return int16_t(p[0] | (p[1] << 8)); }
}  // namespace

bool Bno055::writeReg(uint8_t reg, uint8_t val) {
  for (int attempt = 0; attempt < 5; ++attempt) {
    while (s_.available()) s_.read();
    const uint8_t msg[5] = {0xAA, 0x00, reg, 1, val};
    s_.write(msg, sizeof msg);
    const uint32_t t0 = millis();
    uint8_t resp[2];
    int got = 0;
    while (got < 2 && millis() - t0 < 60) {
      if (s_.available()) resp[got++] = s_.read();
    }
    if (got == 2 && resp[0] == 0xEE && resp[1] == 0x01) return true;
    ++errors_;
    delay(10);
  }
  return false;
}

bool Bno055::readRegs(uint8_t reg, uint8_t n, uint8_t *buf, uint32_t timeoutMs) {
  for (int attempt = 0; attempt < 5; ++attempt) {
    while (s_.available()) s_.read();
    const uint8_t msg[4] = {0xAA, 0x01, reg, n};
    s_.write(msg, sizeof msg);
    const uint32_t t0 = millis();
    int got = 0;
    uint8_t head[2];
    bool fail = false;
    while (millis() - t0 < timeoutMs) {
      if (!s_.available()) continue;
      const uint8_t c = s_.read();
      if (got < 2) {
        head[got++] = c;
        if (got == 2 && (head[0] != 0xBB || head[1] != n)) { fail = true; break; }
      } else {
        buf[got - 2] = c;
        if (++got == n + 2) return true;
      }
    }
    (void)fail;
    ++errors_;
    delay(10);
  }
  return false;
}

bool Bno055::begin(int rxPin, int txPin, uint8_t mode) {
  s_.begin(115200, SERIAL_8N1, rxPin, txPin);
  mode_ = mode;
  delay(700);  // power-on boot time is 650 ms
  // The first frame after power-up is often answered with an error; just send it.
  writeReg(kRegPage, 0);
  uint8_t id = 0;
  if (!readRegs(kRegChipId, 1, &id) || id != kChipId) { ok_ = false; return false; }
  writeReg(kRegOprMode, kModeConfig);
  delay(25);
  writeReg(kRegSysTrigger, 0x20);  // RST_SYS (no reset pin is wired to the ESP32)
  delay(700);
  id = 0;
  for (int i = 0; i < 10 && id != kChipId; ++i) readRegs(kRegChipId, 1, &id);
  if (id != kChipId) { ok_ = false; return false; }
  ok_ = writeReg(kRegPwrMode, 0x00)        // normal power
        && writeReg(kRegPage, 0)
        && writeReg(kRegUnitSel, 0x00)     // m/s^2, deg/s, degrees, Celsius
        && writeReg(kRegSysTrigger, 0x00)  // internal oscillator (no crystal on the board)
        && writeReg(kRegOprMode, mode_);
  delay(30);
  st_ = St::Idle;
  return ok_;
}

void Bno055::decode(const uint8_t *b, ImuRaw &out) const {
  // b[0] is register 0x14
  for (int k = 0; k < 3; ++k) out.gyroDps[k] = le16(b + 0 + 2 * k) / 16.0f;
  for (int k = 0; k < 4; ++k) out.quat[k] = le16(b + 0x0C + 2 * k) / 16384.0f;
  for (int k = 0; k < 3; ++k) out.liaMs2[k] = le16(b + 0x14 + 2 * k) / 100.0f;
  for (int k = 0; k < 3; ++k) out.grvMs2[k] = le16(b + 0x1A + 2 * k) / 100.0f;
  out.calib = b[kBurstLen - 1];
  const float n = sqrtf(out.quat[0] * out.quat[0] + out.quat[1] * out.quat[1] +
                        out.quat[2] * out.quat[2] + out.quat[3] * out.quat[3]);
  if (n > 0.5f)
    for (float &q : out.quat) q /= n;
}

bool Bno055::poll(ImuRaw &out, uint32_t periodUs) {
  if (!ok_) return false;
  const uint32_t now = micros();
  if (st_ == St::Idle) {
    if (now - lastReqUs_ < periodUs) return false;
    while (s_.available()) s_.read();
    const uint8_t msg[4] = {0xAA, 0x01, kBurstReg, kBurstLen};
    s_.write(msg, sizeof msg);
    lastReqUs_ = reqUs_ = now;
    st_ = St::Head;
    return false;
  }
  if (now - reqUs_ > 25000) {  // no complete reply: give up, ask again next period
    ++errors_;
    st_ = St::Idle;
    return false;
  }
  while (s_.available()) {
    const uint8_t c = s_.read();
    switch (st_) {
      case St::Head:
        if (c == 0xBB) st_ = St::Len;
        else if (c == 0xEE) st_ = St::Err;
        break;
      case St::Len:
        if (c != kBurstLen) { ++errors_; st_ = St::Idle; return false; }
        rxPos_ = 0;
        st_ = St::Data;
        break;
      case St::Data:
        rx_[rxPos_++] = c;
        if (rxPos_ == kBurstLen) {
          // The registers are latched roughly when the request finished arriving
          // (4 bytes at 115200 baud ~ 350 us).
          out.tUs = reqUs_ + 350;
          decode(rx_, out);
          st_ = St::Idle;
          return true;
        }
        break;
      case St::Err:  // status byte, e.g. 0x07 bus over-run: just retry next period
        ++errors_;
        st_ = St::Idle;
        return false;
      default:
        st_ = St::Idle;
    }
  }
  return false;
}
