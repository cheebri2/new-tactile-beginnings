#include "flash_log.h"

namespace {
constexpr uint8_t kCmdWren = 0x06, kCmdRdsr1 = 0x05, kCmdRead = 0x03, kCmdPageProgram = 0x02,
                  kCmdChipErase = 0xC7, kCmdJedec = 0x9F, kCmdWake = 0xAB;
const SPISettings kSpi(20000000, MSBFIRST, SPI_MODE0);
}  // namespace

uint8_t FlashLog::status() {
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdRdsr1);
  const uint8_t s = SPI.transfer(0);
  deselect();
  SPI.endTransaction();
  return s;
}

bool FlashLog::waitIdle(uint32_t timeoutMs) {
  const uint32_t t0 = millis();
  while (status() & 0x01) {
    if (millis() - t0 > timeoutMs) return false;
  }
  return true;
}

void FlashLog::writeEnable() {
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdWren);
  deselect();
  SPI.endTransaction();
}

void FlashLog::read(uint32_t addr, uint8_t *buf, size_t n) {
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdRead);
  SPI.transfer((addr >> 16) & 0xFF);
  SPI.transfer((addr >> 8) & 0xFF);
  SPI.transfer(addr & 0xFF);
  for (size_t i = 0; i < n; ++i) buf[i] = SPI.transfer(0);
  deselect();
  SPI.endTransaction();
}

void FlashLog::program(uint32_t addr, const uint8_t *buf, size_t n) {
  writeEnable();
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdPageProgram);
  SPI.transfer((addr >> 16) & 0xFF);
  SPI.transfer((addr >> 8) & 0xFF);
  SPI.transfer(addr & 0xFF);
  for (size_t i = 0; i < n; ++i) SPI.transfer(buf[i]);
  deselect();
  SPI.endTransaction();
  // Programming (up to 3 ms) carries on inside the chip; the next write waits for it.
}

bool FlashLog::begin() {
  pinMode(kCsPin, OUTPUT);
  deselect();
  SPI.begin(kSckPin, kMisoPin, kMosiPin, kCsPin);
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdWake);
  deselect();
  delayMicroseconds(50);
  select();
  SPI.transfer(kCmdJedec);
  jedec_ = uint32_t(SPI.transfer(0)) << 16;
  jedec_ |= uint32_t(SPI.transfer(0)) << 8;
  jedec_ |= SPI.transfer(0);
  deselect();
  SPI.endTransaction();
  // W25Q32JV = EF 40 16. Accept any Winbond-like answer that is not a floating bus.
  ok_ = jedec_ != 0 && jedec_ != 0xFFFFFF;
  if (!ok_) return false;
  waitIdle(30000);

  // Binary search for the first record whose type byte is erased (0xFF).
  uint32_t lo = 0, hi = kSize / sizeof(LogRec);  // answer in [lo, hi]
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    uint8_t t;
    read(mid * sizeof(LogRec), &t, 1);
    if (t == 0xFF) hi = mid; else lo = mid + 1;
  }
  addr_ = lo * sizeof(LogRec);
  bufLen_ = 0;
  return true;
}

void FlashLog::append(LogRec r) {
  if (!ok_ || full()) return;
  uint16_t sum = 0;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(&r);
  for (size_t i = 0; i < offsetof(LogRec, sum); ++i) sum += p[i];
  r.sum = sum;
  if (bufLen_ >= sizeof page_) {  // last page write was refused (chip busy): retry first
    flush();
    if (bufLen_) return;  // still busy: drop this record rather than overrun the buffer
  }
  if (bufLen_ == 0) bufStart_ = addr_;
  memcpy(page_ + bufLen_, &r, sizeof r);
  bufLen_ += sizeof r;
  addr_ += sizeof r;
  if (addr_ % 256 == 0) flush();  // records never straddle a page (32 divides 256)
}

void FlashLog::flush() {
  if (!ok_ || bufLen_ == 0) return;
  if (!waitIdle(10)) return;  // previous page still programming; try again later
  // A partial page is fine: the next records start a new buffer at the next free address
  // in the same page, and the page is only ever programmed over still-erased bytes.
  program(bufStart_, page_, bufLen_);
  bufLen_ = 0;
}

bool FlashLog::eraseAll(Print &progress) {
  if (!ok_) return false;
  bufLen_ = 0;
  waitIdle(30000);
  writeEnable();
  SPI.beginTransaction(kSpi);
  select();
  SPI.transfer(kCmdChipErase);
  deselect();
  SPI.endTransaction();
  const uint32_t t0 = millis();
  while (status() & 0x01) {
    if (millis() - t0 > 60000) return false;
    delay(500);
    progress.print('.');
  }
  addr_ = 0;
  return true;
}

void FlashLog::dump(Print &out) {
  flush();
  waitIdle(50);
  uint8_t buf[256];
  for (uint32_t a = 0; a < addr_; a += sizeof buf) {
    const size_t n = min<uint32_t>(sizeof buf, addr_ - a);
    read(a, buf, n);
    out.write(buf, n);
  }
}
