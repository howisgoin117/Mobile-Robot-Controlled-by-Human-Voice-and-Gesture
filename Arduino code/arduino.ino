/*
 * ODrive Differential Drive — Axis0=TRAI, Axis1=PHAI (nguoc)
 * diffbot_agv_preset — Fixed speed 0.5 + joystick chỉ lái (AGV mode)
 * v2.1 — Fix: String→char[], ODrive timeout, ODrive error check,
 *         PS2 reconnect, non-blocking serial read.
 *
 * Điều khiển:
 * Dpad UP      = tiến (vel = 0.5)
 * Dpad DOWN    = lùi  (vel = 0.5)
 * L-stick X    = vi sai trái/phải
 * R2           = Enable / ESTOP
 * Cross        = ESTOP
 * L1           = Brake
 * Tam giac     = Serial mode
 * Vuong        = PS2 mode
 *
 * Serial: e d s b f stop | v <fwd> [steer]
 */

#include <HardwareSerial.h>
#include <ODriveArduino.h>
#include <avr/wdt.h>
#include <PS2X_lib.h>

template<class T> inline Print& operator<<(Print& o, T a)     { o.print(a);    return o; }
template<>        inline Print& operator<<(Print& o, float a)  { o.print(a, 4); return o; }

HardwareSerial& odrv_ser = Serial1;
ODriveArduino   odrive(odrv_ser);

// ── Pins ──────────────────────────────────────────────────────────────────────
#define PS2_DAT 50
#define PS2_CMD 51
#define PS2_SEL 53
#define PS2_CLK 52
#define ESTOP_PIN   2
#define ULTRA_TRIG  5
#define ULTRA_ECHO  6

// ── Tuning ────────────────────────────────────────────────────────────────────
#define MAX_VEL       3.0f
#define FIXED_VEL     0.5f
#define DEADZONE      25
#define STEER_RATIO   0.6f
#define SAFE_CM       50.0f

// ── Intervals (ms) ────────────────────────────────────────────────────────────
#define T_SEND        40
#define T_PRINT       2000
#define T_WDT         80
#define T_ULTRA       120
#define T_DEBOUNCE    200
#define T_PS2_READ    20
#define T_ODRV_CHECK  3000
#define T_PS2_RETRY   5000

// ── Serial buffer ─────────────────────────────────────────────────────────────
#define SER_BUF_SIZE  64

// ── State ─────────────────────────────────────────────────────────────────────
PS2X ps2x;
bool ps2Ready     = false;
bool motorOn      = false;
bool ultraBlocked = false;
bool debugPS2     = false;

enum AppMode { MODE_PS2, MODE_SER } mode = MODE_PS2;

float tgtL=0, tgtR=0;
float prvSendL=0, prvSendR=0;

unsigned long tWdt=0, tSend=0, tPrint=0, tUltra=0;
unsigned long tPS2Read=0, tOdrvCheck=0, tPS2Retry=0;
unsigned long tCross=0, tR2=0, tTri=0, tSqr=0;
bool wCross=false, wR2=false, wTri=false, wSqr=false;

// Serial read buffer (non-blocking)
char serBuf[SER_BUF_SIZE];
uint8_t serIdx = 0;

volatile bool estopFlag = false;
volatile char estopSrc  = ' ';

// ── ISR ───────────────────────────────────────────────────────────────────────
void estopISR() { estopFlag=true; estopSrc='B'; }

// ── Helpers ───────────────────────────────────────────────────────────────────
void sendVel(float l, float r) {
  odrive.SetVelocity(0, l);
  odrive.SetVelocity(1, r);
}

float readUltra() {
  digitalWrite(ULTRA_TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(ULTRA_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(ULTRA_TRIG, LOW);
  long d = pulseIn(ULTRA_ECHO, HIGH, 25000);
  return d ? d*0.017f : 999.0f;
}

float joyAxis(byte raw) {
  int c = (int)raw - 128;
  if (abs(c) < DEADZONE) return 0.0f;
  float n = constrain((float)(abs(c)-DEADZONE)/(128.0f-DEADZONE), 0.0f, 1.0f);
  return (c>0 ? 1.0f : -1.0f) * n;
}

void enableMotors() {
  odrv_ser << "sc\n";
  wdt_reset(); delay(50); wdt_reset();
  bool ok0 = odrive.run_state(0, AXIS_STATE_CLOSED_LOOP_CONTROL, false); wdt_reset();
  bool ok1 = odrive.run_state(1, AXIS_STATE_CLOSED_LOOP_CONTROL, false); wdt_reset();
  if (ok0&&ok1) { motorOn=true; ultraBlocked=false; Serial.println(F("[OK] Motor ON")); }
  else          { Serial.println(F("[ERR] Enable FAIL - goi 's'")); }
}

void emergencyStop(const char* r) {
  Serial.print(F("[ESTOP] ")); Serial.println(r);
  tgtL=tgtR=0;
  sendVel(0,0);
  wdt_reset(); delay(80); wdt_reset();
  odrive.run_state(0, AXIS_STATE_IDLE, false); wdt_reset();
  odrive.run_state(1, AXIS_STATE_IDLE, false); wdt_reset();
  motorOn=false; ultraBlocked=false;
}

void softBrake(const char* r) {
  tgtL=tgtR=0;
  Serial.print(F("[BRAKE] ")); Serial.println(r);
}

// ── Emergency + Ultrasonic ────────────────────────────────────────────────────
void checkEmergency() {
  if (estopFlag) {
    estopFlag=false;
    emergencyStop(estopSrc=='B' ? "Nut bam" : "PS2");
    return;
  }
  if (millis()-tUltra < T_ULTRA) return;
  tUltra=millis();
  wdt_reset(); float d=readUltra(); wdt_reset();
  if (d<SAFE_CM && motorOn && !ultraBlocked) {
    ultraBlocked=true;
    Serial.print(F("[ULTRA] ")); Serial.print(d); Serial.println(F("cm BLOCK tien"));
  } else if (d>=SAFE_CM && ultraBlocked) {
    ultraBlocked=false;
    Serial.println(F("[ULTRA] OK"));
  }
}

// ── ODrive error check ───────────────────────────────────────────────────────
void checkODriveErrors() {
  if (millis()-tOdrvCheck < T_ODRV_CHECK) return;
  tOdrvCheck=millis();
  if (!motorOn) return;

  odrv_ser.setTimeout(200);

  odrv_ser << "r axis0.error\n";
  wdt_reset();
  long err0 = odrive.readInt();
  wdt_reset();

  odrv_ser << "r axis1.error\n";
  long err1 = odrive.readInt();
  wdt_reset();

  odrv_ser.setTimeout(1000);

  if (err0 != 0 || err1 != 0) {
    Serial.print(F("[ODRV ERR] ax0=")); Serial.print(err0);
    Serial.print(F(" ax1=")); Serial.println(err1);
    emergencyStop("ODrive error");
  }
}

// ── PS2 ───────────────────────────────────────────────────────────────────────
void handlePS2() {
  // Retry kết nối PS2 nếu chưa sẵn sàng
  if (!ps2Ready) {
    if (millis()-tPS2Retry >= T_PS2_RETRY) {
      tPS2Retry=millis();
      wdt_reset();
      int err=ps2x.config_gamepad(PS2_CLK,PS2_CMD,PS2_SEL,PS2_DAT,false,false);
      wdt_reset();
      ps2Ready=(err==0);
      if (ps2Ready) Serial.println(F("[PS2] Reconnected!"));
    }
    return;
  }

  if (millis()-tPS2Read < T_PS2_READ) return;
  tPS2Read=millis();

  bool ok = ps2x.read_gamepad(false, 0);
  if (!ok) return; // Bỏ qua gói tin lỗi, không làm gì thêm

  // Đọc analog 1 lần duy nhất
  byte lx = ps2x.Analog(PSS_LX);

  if (debugPS2) {
    byte ly = ps2x.Analog(PSS_LY);
    byte rx = ps2x.Analog(PSS_RX); byte ry = ps2x.Analog(PSS_RY);
    Serial.print(F("[PS2 DEBUG] UP=")); Serial.print(ps2x.Button(PSB_PAD_UP));
    Serial.print(F(" DOWN=")); Serial.print(ps2x.Button(PSB_PAD_DOWN));
    Serial.print(F(" | L(")); Serial.print(lx); Serial.print(F(",")); Serial.print(ly);
    Serial.print(F(") R(")); Serial.print(rx); Serial.print(F(",")); Serial.println(ry);
  }

  // ── CROSS = ESTOP ──────────────────────────────────────────────────────────
  bool cn=ps2x.Button(PSB_CROSS);
  if (cn&&!wCross&&millis()-tCross>=T_DEBOUNCE) {
    tCross=millis(); wCross=true; estopSrc='P'; estopFlag=true; return;
  }
  if (!cn) wCross=false;

  // ── R2 = Enable / ESTOP ───────────────────────────────────────────────────
  bool r2=ps2x.Button(PSB_R2);
  if (r2&&!wR2&&millis()-tR2>=T_DEBOUNCE) {
    tR2=millis(); wR2=true;
    if (!motorOn) enableMotors(); else { estopSrc='P'; estopFlag=true; }
    return;
  }
  if (!r2) wR2=false;

  // ── TAM GIAC = Serial mode ────────────────────────────────────────────────
  bool tri=ps2x.Button(PSB_TRIANGLE);
  if (tri&&!wTri&&millis()-tTri>=T_DEBOUNCE) {
    tTri=millis(); wTri=true;
    if (mode!=MODE_SER) { mode=MODE_SER; softBrake("->SER"); }
  }
  if (!tri) wTri=false;

  // ── VUONG = PS2 mode ──────────────────────────────────────────────────────
  bool sq=ps2x.Button(PSB_SQUARE);
  if (sq&&!wSqr&&millis()-tSqr>=T_DEBOUNCE) {
    tSqr=millis(); wSqr=true;
    if (mode!=MODE_PS2) { mode=MODE_PS2; softBrake("->PS2"); }
  }
  if (!sq) wSqr=false;

  // ── L1 = Brake ────────────────────────────────────────────────────────────
  if (ps2x.Button(PSB_L1)) { softBrake("L1"); return; }

  // ── Dpad + L-stick → target velocity (chỉ khi MODE_PS2) ─────────────────
  if (mode==MODE_PS2) {
    float thrAxis = 0;
    if      (ps2x.Button(PSB_PAD_UP))   thrAxis =  1.0f;
    else if (ps2x.Button(PSB_PAD_DOWN)) thrAxis = -1.0f;
    else { tgtL=tgtR=0; return; } // Không nhấn Dpad → dừng

    float steerAxis = joyAxis(lx);
    float baseVel   = thrAxis * FIXED_VEL;
    float steerVel  = steerAxis * FIXED_VEL * STEER_RATIO;

    tgtL = constrain( baseVel - steerVel, -MAX_VEL, MAX_VEL);
    tgtR = constrain(-(baseVel + steerVel), -MAX_VEL, MAX_VEL);
  }
}

// ── Serial command processing ─────────────────────────────────────────────────
void processCommand(const char* cmd) {
  if      (strcmp(cmd,"e")==0)                       { if (!motorOn) enableMotors(); else emergencyStop("Serial"); }
  else if (strcmp(cmd,"d")==0)                       { emergencyStop("User"); }
  else if (strcmp(cmd,"stop")==0||strcmp(cmd,"brake")==0) { softBrake("Serial"); }
  else if (strcmp(cmd,"s")==0)                       { odrv_ser<<"sc\n"; wdt_reset(); Serial.println(F("[OK] Cleared")); }
  else if (strcmp(cmd,"b")==0) {
    odrv_ser<<"r vbus_voltage\n"; wdt_reset();
    odrv_ser.setTimeout(200);
    float v = odrive.readFloat();
    wdt_reset();
    odrv_ser.setTimeout(1000);
    Serial.print(F("Vbus=")); Serial.println(v);
  }
  else if (strcmp(cmd,"f")==0) {
    Serial.print(F("L=")); Serial.print(tgtL,2);
    Serial.print(F(" R=")); Serial.println(tgtR,2);
  }
  else if (strcmp(cmd,"ps2")==0) {
    debugPS2 = !debugPS2;
    if (debugPS2) Serial.println(F("[DEBUG] PS2 ON - Nhap lai 'ps2' de tat"));
    else          Serial.println(F("[DEBUG] PS2 OFF"));
  }
  else if (strncmp(cmd,"v ",2)==0) {
    if (mode!=MODE_SER) { Serial.println(F("[ERR] Mode PS2")); return; }
    if (!motorOn)       { Serial.println(F("[ERR] Motor off")); return; }
    const char* arg = cmd + 2;
    while (*arg==' ') arg++; // trim leading spaces
    float fwd=0, str=0;
    char* endp;
    fwd = strtod(arg, &endp);
    fwd = constrain(fwd, -MAX_VEL, MAX_VEL);
    if (*endp==' ') {
      str = strtod(endp, NULL);
      str = constrain(str, -MAX_VEL, MAX_VEL);
    }
    tgtL=constrain( fwd+str,-MAX_VEL,MAX_VEL);
    tgtR=constrain(-(fwd-str),-MAX_VEL,MAX_VEL);
    Serial.print(F("[OK] L=")); Serial.print(tgtL,2);
    Serial.print(F(" R=")); Serial.print(tgtR,2);
    if (ultraBlocked) Serial.println(F(" [pending]")); else Serial.println();
  }
  else { Serial.println(F("[?] e d s b f ps2 stop | v <fwd> [steer]")); }
}

// ── Serial read (non-blocking) ───────────────────────────────────────────────
void handleSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c=='\n' || c=='\r') {
      if (serIdx==0) continue;       // bỏ qua dòng trống
      serBuf[serIdx] = '\0';
      serIdx = 0;
      // trim trailing spaces
      uint8_t len = strlen(serBuf);
      while (len>0 && serBuf[len-1]==' ') { serBuf[--len]='\0'; }
      if (len>0) processCommand(serBuf);
      return;
    }
    if (serIdx < SER_BUF_SIZE-1) {
      serBuf[serIdx++] = c;
    }
    // overflow: bỏ ký tự thừa, chờ \n
  }
}

// ── Motor control ─────────────────────────────────────────────────────────────
void updateMotorControl() {
  if (!motorOn) return;
  if (millis()-tSend < T_SEND) return;
  tSend=millis();

  float sendL = tgtL;
  float sendR = tgtR;

  // Vật cản: chỉ chặn tiến, cho phép lùi
  if (ultraBlocked) {
    if (sendL > 0) sendL = 0;
    if (sendR < 0) sendR = 0;
  }
  sendVel(sendL, sendR);

  // Log khi có thay đổi hoặc định kỳ (so sánh giá trị thực gửi)
  bool justStopped = (abs(prvSendL)>0.05f||abs(prvSendR)>0.05f) && (abs(sendL)<0.05f&&abs(sendR)<0.05f);
  bool changed     = abs(sendL-prvSendL)>0.05f || abs(sendR-prvSendR)>0.05f;
  bool periodic    = (millis()-tPrint>=T_PRINT) && (abs(sendL)>0.05f||abs(sendR)>0.05f);

  if (justStopped||changed||periodic) {
    tPrint=millis(); prvSendL=sendL; prvSendR=sendR;
    Serial.print(mode==MODE_PS2?F("[PS2]"):F("[SER]"));
    Serial.print(F(" L=")); Serial.print(sendL,1);
    Serial.print(F(" R=")); Serial.print(sendR,1);
    Serial.print(F(" tgt=")); Serial.print(tgtL,1);
    Serial.print(F("/")); Serial.print(tgtR,1);
    if (ultraBlocked) Serial.println(F(" [BLK-tien]"));
    else              Serial.println();
  }
}

// ── Setup / Loop ──────────────────────────────────────────────────────────────
void setup() {
  wdt_disable();
  odrv_ser.begin(19200);
  Serial.begin(115200);
  // Không cần while(!Serial) trên Mega — UART0 luôn sẵn sàng

  pinMode(ESTOP_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ESTOP_PIN), estopISR, FALLING);
  pinMode(ULTRA_TRIG, OUTPUT);
  pinMode(ULTRA_ECHO, INPUT);

  odrv_ser<<"sc\n"; wdt_reset(); delay(200); wdt_reset();
  odrv_ser<<"w axis0.controller.config.vel_limit "<<MAX_VEL<<'\n';
  odrv_ser<<"w axis0.motor.config.current_lim 4.0\n";
  odrv_ser<<"w axis1.controller.config.vel_limit "<<MAX_VEL<<'\n';
  odrv_ser<<"w axis1.motor.config.current_lim 4.0\n";
  wdt_reset(); delay(200); wdt_reset();

  int err=ps2x.config_gamepad(PS2_CLK,PS2_CMD,PS2_SEL,PS2_DAT,false,false);
  ps2Ready=(err==0);
  Serial.println(ps2Ready?F("[PS2] OK"):F("[PS2] Not found - auto retry"));
  Serial.println(F("[BOOT] AGV v2.1 Ready."));
  Serial.println(F("  Dpad UP=tien  Dpad DOWN=lui  L-stick=vi sai"));
  Serial.println(F("  R2=enable  Cross=estop  L1=brake"));
  Serial.println(F("  Serial: e d s b f ps2 stop | v <fwd> [steer]"));
  Serial.print(F("  Fixed vel = ")); Serial.println(FIXED_VEL);
  wdt_enable(WDTO_500MS);
}

void loop() {
  if (millis()-tWdt>=T_WDT) { wdt_reset(); tWdt=millis(); }
  checkEmergency();
  checkODriveErrors();
  handlePS2();
  handleSerial();
  updateMotorControl();
}
