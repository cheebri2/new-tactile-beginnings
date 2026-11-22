// Append-only log on the W25Q32JVSS SPI flash (4 MB), a backup of the IMU track and
// sweep events that survives a lost USB link or a power cut.
//   CS = GPIO5 (10k pull-up), CLK = GPIO18, MISO = GPIO19, MOSI = GPIO23 (VSPI defaults)
//   WP and HOLD are pulled up on the board.
//
// 32-byte records, written a page (8 records) at a time. Erased flash reads 0xFF, so the
// first record whose type byte is 0xFF is the end of the log; it is found at boot with a
// binary search. Nothing is erased while logging (erase takes up to 25 s): run
// `flash erase` once before a session. At 25 Hz the chip holds about 87 minutes.
// Read it out with pc/flash_dump.py.

#pragma once
#include <Arduino.h>
#include <SPI.h>

struct __attribute__((packed)) LogRec {
  uint8_t type;     // see LogType
  uint8_t flags;    // Tracker flags
  uint8_t calib;    // BNO055 CALIB_STAT
  uint8_t pad;
  uint32_t tMs;     // millis() since boot
  int16_t yawCdeg, pitchCdeg, rollCdeg;  // 0.01 deg
  int16_t pad2;
  float xMm, yMm;
  float value;      // mark number for marks, 1/0 = finished/aborted for sweep end
  uint16_t reserved;
  uint16_t sum;     // 16-bit sum of bytes 0..29
};
static_assert(sizeof(LogRec) == 32, "LogRec must be 32 bytes");

enum LogType : uint8_t {
  kLogImu = 1, kLogZero = 2, kLogSweepStart = 3, kLogSweepEnd = 4, kLogMark = 5, kLogBoot = 6
};

class FlashLog {
 public:
  static constexpr uint32_t kSize = 4u * 1024 * 1024;
  static constexpr uint8_t kCsPin = 5, kSckPin = 18, kMisoPin = 19, kMosiPin = 23;

  bool begin();                 // finds the end of the log; false if no chip answers
  bool ok() const { return ok_; }
  uint32_t jedec() const { return jedec_; }
  uint32_t used() const { return addr_; }
  bool full() const { return addr_ + sizeof(LogRec) > kSize; }

  void append(LogRec r);        // buffered; programs a page when 8 records are waiting
  void flush();                 // program whatever is buffered (partial page is fine)
  bool eraseAll(Print &progress);  // chip erase, blocking (10-25 s)
  void dump(Print &out);        // raw bytes 0..used()

 private:
  void select() { digitalWrite(kCsPin, LOW); }
  void deselect() { digitalWrite(kCsPin, HIGH); }
  uint8_t status();
  bool waitIdle(uint32_t timeoutMs);
  void writeEnable();
  void read(uint32_t addr, uint8_t *buf, size_t n);
  void program(uint32_t addr, const uint8_t *buf, size_t n);  // within one page

  bool ok_ = false;
  uint32_t jedec_ = 0;
  uint32_t addr_ = 0;           // next free byte
  uint8_t page_[256];
  uint32_t bufStart_ = 0;
  uint16_t bufLen_ = 0;
};
