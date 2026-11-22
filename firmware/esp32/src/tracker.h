// Head pose on the phantom from the BNO055: angle and estimated x/y displacement.
// Mirrored line for line in pc/piezo/imu.py (Tracker), which the tests run on synthetic moves.
//
// Angle: the fused quaternion relative to the pose at `zero` (yaw about vertical, then
// pitch, roll). Good to about a degree; in IMU mode yaw drifts slowly (gyro only).
//
// Position: an accelerometer cannot give position by plain double integration. A bias of
// 0.05 m/s^2 is 2.5 cm of error after 1 s and 2.5 m after 10 s. What makes it usable here
// is that a phantom test is "slide the head, hold still for a sweep, slide again":
//   - Zero-velocity updates: whenever the head is still (or a sweep is playing, when it
//     must be still), velocity is forced to 0 and position is frozen.
//   - Each slide is one segment. Velocity must be 0 at both ends, so the velocity left
//     over at the end is drift; it is removed as a linear ramp over the segment
//     (position correction = v_end * T / 2).
//   - Accelerometer bias is re-estimated (sensor frame) during every still period.
// Expect centimetre-level, not millimetre-level, accuracy, and worse for slow creeping
// moves (constant velocity looks "still"). Quick slides of 0.3-3 s work best. The x/y
// you type in capture.py stays the ground truth; the IMU estimate is saved next to it.
//
// Frames: x/y are in the phantom plane, origin where `zero` was done (lower-left corner).
//   x = horizontal direction of the board's x axis at zero; y = 90 deg anticlockwise from
//   x seen from above. Remap with --imu-axes on the laptop if the board is mounted rotated.

#pragma once
#include <math.h>
#include <stdint.h>

struct TrackerOut {
  float yawDeg, pitchDeg, rollDeg;  // relative to the zero pose (absolute before zero)
  float xMm, yMm;                   // NAN before zero
  float axMs2, ayMs2;               // phantom-frame acceleration, bias removed
  uint8_t flags;
};

class Tracker {
 public:
  enum Flag : uint8_t {
    kZeroed = 1, kMoving = 2, kDriveHold = 4, kLongMove = 8, kGravityOdd = 16, kStale = 32
  };
  // Tunables (keep in step with pc/piezo/imu.py)
  float stillGyroDps = 3.0f;    // |gyro| below this ...
  float stillAccMs2 = 0.25f;    // ... and |acc - bias| below this ...
  uint16_t stillSamples = 25;   // ... for this many samples (0.25 s at 100 Hz) = still
  float biasAlpha = 0.02f;      // bias low-pass while still
  float longMoveS = 5.0f;       // segments longer than this are flagged (drift grows as T^2)

  // Ask to zero at the next confirmed-still sample. Returns via zeroDone().
  void requestZero() { zeroPending_ = true; zeroDone_ = false; }
  bool zeroPending() const { return zeroPending_; }
  bool zeroDone() { bool d = zeroDone_; zeroDone_ = false; return d; }
  bool zeroed() const { return zeroed_; }

  // One IMU sample. q = w,x,y,z sensor->world; lia/grv sensor frame m/s^2; gyro deg/s.
  // driveOn = the amp is playing (the protocol says the head is still during a sweep).
  TrackerOut update(uint32_t tUs, const float q[4], const float lia[3], const float grv[3],
                    const float gyro[3], bool driveOn) {
    float dt = haveT_ ? (int32_t(tUs - lastT_) * 1e-6f) : 0.0f;
    if (dt < 0 || dt > 0.05f) dt = 0.0f;  // gap or duplicate: skip integration this step
    lastT_ = tUs;
    haveT_ = true;

    // Still detection
    const float gyroN = sqrtf(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
    float a[3] = {lia[0] - bias_[0], lia[1] - bias_[1], lia[2] - bias_[2]};
    const float accN = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const bool quiet = gyroN < stillGyroDps && accN < stillAccMs2;
    stillCount_ = quiet ? (stillCount_ < 60000 ? stillCount_ + 1 : stillCount_) : 0;
    const bool still = driveOn || stillCount_ >= stillSamples;

    if (still && !driveOn) {
      for (int k = 0; k < 3; ++k) bias_[k] += biasAlpha * (lia[k] - bias_[k]);
    }

    if (zeroPending_ && stillCount_ >= stillSamples) {
      for (int k = 0; k < 4; ++k) q0_[k] = q[k];
      float xw[3];
      const float ex[3] = {1, 0, 0};
      rotate(q, ex, xw);
      psi0_ = atan2f(xw[1], xw[0]);
      float gw[3];
      rotate(q, grv, gw);
      const float gN = sqrtf(grv[0] * grv[0] + grv[1] * grv[1] + grv[2] * grv[2]);
      gravityOdd_ = !(gN > 5.0f && gw[2] > 0.9f * gN);  // world z should be along gravity
      pos_[0] = pos_[1] = 0;
      vel_[0] = vel_[1] = seg_[0] = seg_[1] = 0;
      segT_ = 0;
      moving_ = false;
      longMove_ = false;
      zeroed_ = true;
      zeroPending_ = false;
      zeroDone_ = true;
    }

    // Phantom-frame acceleration: sensor -> world (q), then rotate by -psi0 about vertical.
    float aw[3];
    rotate(q, a, aw);
    const float c = cosf(-psi0_), s = sinf(-psi0_);
    const float ax = c * aw[0] - s * aw[1], ay = s * aw[0] + c * aw[1];

    if (zeroed_) {
      if (!still) {
        if (!moving_) {  // a slide starts
          moving_ = true;
          vel_[0] = vel_[1] = seg_[0] = seg_[1] = 0;
          segT_ = 0;
        }
        integrate(ax, ay, dt);
      } else if (moving_) {
        // The confirming still samples were integrated as moving; close the segment and
        // remove the leftover velocity as a linear drift over it.
        integrate(ax, ay, dt);
        pos_[0] += seg_[0] - vel_[0] * segT_ / 2;
        pos_[1] += seg_[1] - vel_[1] * segT_ / 2;
        longMove_ = segT_ > longMoveS;
        moving_ = false;
        vel_[0] = vel_[1] = seg_[0] = seg_[1] = 0;
        segT_ = 0;
      }
    }

    TrackerOut o;
    float qr[4];
    if (zeroed_) {
      const float q0c[4] = {q0_[0], -q0_[1], -q0_[2], -q0_[3]};
      mul(q0c, q, qr);
    } else {
      for (int k = 0; k < 4; ++k) qr[k] = q[k];
    }
    euler(qr, o.yawDeg, o.pitchDeg, o.rollDeg);
    o.xMm = zeroed_ ? 1000.0f * (pos_[0] + (moving_ ? seg_[0] : 0)) : NAN;
    o.yMm = zeroed_ ? 1000.0f * (pos_[1] + (moving_ ? seg_[1] : 0)) : NAN;
    o.axMs2 = ax;
    o.ayMs2 = ay;
    o.flags = (zeroed_ ? kZeroed : 0) | (moving_ ? kMoving : 0) | (driveOn ? kDriveHold : 0) |
              (longMove_ ? kLongMove : 0) | (gravityOdd_ ? kGravityOdd : 0);
    return o;
  }

 private:
  void integrate(float ax, float ay, float dt) {
    vel_[0] += ax * dt;
    vel_[1] += ay * dt;
    seg_[0] += vel_[0] * dt;
    seg_[1] += vel_[1] * dt;
    segT_ += dt;
  }
  static void mul(const float a[4], const float b[4], float r[4]) {
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  }
  // v_world = q * v * q^-1
  static void rotate(const float q[4], const float v[3], float out[3]) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    out[0] = (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2];
    out[1] = 2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2];
    out[2] = 2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2];
  }
  // ZYX: yaw about z (anticlockwise from above = positive), then pitch about y, roll about x.
  static void euler(const float q[4], float &yaw, float &pitch, float &roll) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    constexpr float kDeg = 57.29577951f;
    yaw = atan2f(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * kDeg;
    float sp = 2 * (w * y - z * x);
    sp = sp > 1 ? 1 : (sp < -1 ? -1 : sp);
    pitch = asinf(sp) * kDeg;
    roll = atan2f(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * kDeg;
  }

  bool zeroPending_ = false, zeroDone_ = false, zeroed_ = false, moving_ = false;
  bool longMove_ = false, gravityOdd_ = false, haveT_ = false;
  uint32_t lastT_ = 0;
  uint16_t stillCount_ = 0;
  float q0_[4] = {1, 0, 0, 0}, psi0_ = 0;
  float bias_[3] = {0, 0, 0};
  float pos_[2] = {0, 0}, vel_[2] = {0, 0}, seg_[2] = {0, 0}, segT_ = 0;
};
