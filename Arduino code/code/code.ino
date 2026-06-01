#include <HardwareSerial.h>
#include <ODriveArduino.h>
#include <avr/wdt.h>
#include <PS2X_lib.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include <Servo.h>

int servoPin1 = 30;
Servo hori_serv;

int servoPin2 = 31;
Servo verti_serv;

// Speeds
int pan_left_speed = 80;  
int pan_right_speed = 100; 
int tilt_up_speed = 100;   
int tilt_down_speed = 80;  

// --- NEW: Watchdog Timer Variables ---
// This controls how long the servo moves per command (in milliseconds)
// 150ms is roughly equivalent to a small 20-30 degree "step".
const unsigned long burst_duration = 150; 

unsigned long last_pan_time = 0;
unsigned long last_tilt_time = 0;

bool is_panning = false;
bool is_tilting = false;


template<class T> inline Print& operator<<(Print& o, T a)    { o.print(a);    return o; }
template<>        inline Print& operator<<(Print& o, float a) { o.print(a, 4); return o; }

HardwareSerial& odrv_ser = Serial1;
ODriveArduino   odrive(odrv_ser);

LiquidCrystal_I2C lcd(0x27, 16, 2);
#define T_LCD  200
unsigned long tLcd = 0;

#define PS2_DAT    50
#define PS2_CMD    51
#define PS2_SEL    53
#define PS2_CLK    52
#define RELAY_PIN  22

#define RELAY_TURN_LEFT  23
#define RELAY_TURN_RIGHT 24
#define TURN_RELAY_ON  HIGH
#define TURN_RELAY_OFF LOW

#define ULTRA1_TRIG 5
#define ULTRA1_ECHO 3

#define ULTRA2_TRIG 4
#define ULTRA2_ECHO 2

#define MAX_VEL       3.0f
#define DEADZONE      25
#define STEER_RATIO   0.6f
#define FIXED_VEL     0.3f

#define SAFE_CM        50.0f
#define CLEAR_CM       65.0f
#define BLOCK_CONFIRM  3
#define CLEAR_CONFIRM  5
#define RELAY_HOLD_MS  500

#define T_SEND      15
#define T_PRINT     2000
#define T_WDT       80
#define T_ULTRA     60
#define T_DEBOUNCE  200
#define T_PS2_READ  20

#define ODRIVE_READ_TIMEOUT_MS  200

const int LINE_PINS[8] = {A0, A1, A2, A3, A4, A5, A6, A7};
int LINE_THRESHOLD[8]  = {921, 885, 883, 888, 891, 888, 884, 892};
const float LINE_X[8]  = {-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f};

int calibWhite[8]      = {0,0,0,0,0,0,0,0};
int calibBlack[8]      = {0,0,0,0,0,0,0,0};
bool calibWhiteDone    = false;
bool calibBlackDone    = false;

const int LINE_PINS_REAR[8]  = {A8, A9, A10, A11, A12, A13, A14, A15};
int LINE_THRESHOLD_REAR[8]   = {849 ,872 ,886, 885, 868, 888, 859, 849};
const float LINE_X_REAR[8]   = {-3.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.5f};

int calibWhiteRear[8]  = {0,0,0,0,0,0,0,0};
int calibBlackRear[8]  = {0,0,0,0,0,0,0,0};
bool calibWhiteRearDone = false;
bool calibBlackRearDone = false;

int WHITE_ZONE_THRESHOLD[8] = {859, 804, 801, 806, 806, 785, 766, 746};
int WHITE_ZONE_THRESHOLD_REAR[8] = {693, 749, 770, 771, 732, 819, 715, 707};

#define WHITE_ZONE_MIN_COUNT  5
#define WHITE_ZONE_CONFIRM    4
#define WHITE_ZONE_MARGIN     30

int calibWhiteZone[8]      = {0,0,0,0,0,0,0,0};
bool calibWhiteZoneDone    = false;
int calibWhiteZoneRear[8] = {0,0,0,0,0,0,0,0};
bool calibWhiteZoneRearDone = false;

#define REVERSE_BRAKE_MS  800

enum LineDir { DIR_FORWARD, DIR_BRAKING, DIR_REVERSE } lineDir = DIR_FORWARD;
LineDir dirBeforeBrake = DIR_FORWARD;

unsigned long tBrakeStart   = 0;
uint8_t       whiteZoneCount = 0;

float lineKp = 0.35f;
float lineKi = 0.0f;
float lineKd = 0.25f;
float lineSteerAlpha      = 0.35f;
float lineSteerRateLimit = 0.25f;

float linePrevPos     = 0;
float lineIntegral    = 0;
float lineLastPos     = 0;
float lineSteerSmooth = 0;
float lineSteerPrev   = 0;

float LINE_MAX_VEL = FIXED_VEL;
#define LINE_MAX_STEER  (LINE_MAX_VEL * 0.5f)

int lineRead(int i) { return analogRead(LINE_PINS[i]) > LINE_THRESHOLD[i] ? 1 : 0; }
int lineReadRear(int i) { return analogRead(LINE_PINS_REAR[i]) > LINE_THRESHOLD_REAR[i] ? 1 : 0; }

int countWhiteZoneFront() {
  int cnt = 0;
  for (int i = 0; i < 8; i++) { if (analogRead(LINE_PINS[i]) <= WHITE_ZONE_THRESHOLD[i]) cnt++; }
  return cnt;
}

int countWhiteZoneRear() {
  int cnt = 0;
  for (int i = 0; i < 8; i++) { if (analogRead(LINE_PINS_REAR[i]) <= WHITE_ZONE_THRESHOLD_REAR[i]) cnt++; }
  return cnt;
}

float lineGetPosition(bool& lost) {
  float ws  = 0;
  int   cnt = 0;
  for (int i = 0; i < 8; i++) {
    int v = lineRead(i);
    ws  += LINE_X[i] * v;
    cnt += v;
  }
  if (cnt == 0) { lost = true; return lineLastPos; }
  lost = false;
  lineLastPos = ws / cnt;
  return lineLastPos;
}

float lineRearLastPos = 0;
float lineGetPositionRear(bool& lost) {
  float ws  = 0;
  int   cnt = 0;
  for (int i = 0; i < 8; i++) {
    int v = lineReadRear(i);
    ws  += LINE_X_REAR[i] * v;
    cnt += v;
  }
  if (cnt == 0) { lost = true; return lineRearLastPos; }
  lost = false;
  lineRearLastPos = ws / cnt;
  return lineRearLastPos;
}

float lineComputePID(float pos) {
  if (lineKi > 0.001f) {
    lineIntegral += pos;
    lineIntegral  = constrain(lineIntegral, -15.0f, 15.0f);
  } else {
    lineIntegral = 0;
  }

  float deriv = -(pos - linePrevPos);
  linePrevPos = pos;

  float raw = lineKp * pos + lineKi * lineIntegral + lineKd * deriv;
  raw = constrain(raw, -LINE_MAX_STEER, LINE_MAX_STEER);

  if (abs(pos) < 0.3f) {
    lineSteerSmooth *= 0.85f;
  } else {
    lineSteerSmooth = lineSteerAlpha * raw + (1.0f - lineSteerAlpha) * lineSteerSmooth;
  }

  float delta = lineSteerSmooth - lineSteerPrev;
  delta = constrain(delta, -lineSteerRateLimit, lineSteerRateLimit);
  lineSteerPrev = lineSteerPrev + delta;

  return lineSteerPrev;
}

void lineResetPID() {
  linePrevPos = 0; lineIntegral = 0; lineLastPos = 0;
  lineSteerSmooth = 0; lineSteerPrev = 0; lineRearLastPos = 0;
  lineDir       = DIR_FORWARD;
  whiteZoneCount = 0;
}

void printLinePID() {
  Serial.print(F("[LINE PID] Kp=")); Serial.print(lineKp, 3);
  Serial.print(F(" Ki="));          Serial.print(lineKi, 3);
  Serial.print(F(" Kd="));          Serial.print(lineKd, 3);
  Serial.print(F(" Alpha="));       Serial.print(lineSteerAlpha, 3);
  Serial.print(F(" RateLimit="));   Serial.println(lineSteerRateLimit, 3);
}

void printLineSensor() {
  int adc[8];
  for (int i = 0; i < 8; i++) adc[i] = analogRead(LINE_PINS[i]);

  Serial.println(F("-----------------------------------------"));
  Serial.println(F("[LINE TRUOC A0-A7]"));
  Serial.println(F("  S0    S1    S2    S3    S4    S5    S6    S7"));
  Serial.print(F("  ADC   :"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", adc[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  Nguong:"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", LINE_THRESHOLD[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  WZone :"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", WHITE_ZONE_THRESHOLD[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  ON/OFF:  "));
  for (int i = 0; i < 8; i++) { Serial.print(adc[i] > LINE_THRESHOLD[i] ? F("1") : F("0")); if (i < 7) Serial.print(F("     ")); }
  Serial.println();
  
  Serial.print(F("  WZone :  "));
  for (int i = 0; i < 8; i++) { Serial.print(adc[i] <= WHITE_ZONE_THRESHOLD[i] ? F("W") : F(".")); if (i < 7) Serial.print(F("     ")); }
  Serial.println();

  bool lost;
  float pos = lineGetPosition(lost);
  Serial.print(F("  Vi tri: "));
  if (lost) Serial.println(F("MAT LINE"));
  else {
    Serial.print(pos, 3); Serial.print(F("  "));
    if      (abs(pos) < 0.3f) Serial.println(F("THANG"));
    else if (pos < 0)         Serial.println(F("lech TRAI"));
    else                      Serial.println(F("lech PHAI"));
  }
  int wz = countWhiteZoneFront();
  Serial.print(F("  WhiteZone (TRUOC): ")); Serial.print(wz); Serial.print(F("/")); Serial.print(WHITE_ZONE_MIN_COUNT);
  Serial.println(wz >= WHITE_ZONE_MIN_COUNT ? F(" -> DAO CHIEU!") : F(" (binh thuong)"));
  Serial.println(F("-----------------------------------------"));
}

void printLineSensorRear() {
  int adc[8];
  for (int i = 0; i < 8; i++) adc[i] = analogRead(LINE_PINS_REAR[i]);

  Serial.println(F("-----------------------------------------"));
  Serial.println(F("[LINE SAU A8-A15]  (A8=PHAI, A15=TRAI)"));
  Serial.println(F("  S8    S9   S10   S11   S12   S13   S14   S15"));
  Serial.print(F("  ADC   :"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", adc[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  Nguong:"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", LINE_THRESHOLD_REAR[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  WZone :"));
  for (int i = 0; i < 8; i++) { char buf[6]; sprintf(buf, "%6d", WHITE_ZONE_THRESHOLD_REAR[i]); Serial.print(buf); }
  Serial.println();
  Serial.print(F("  ON/OFF:  "));
  for (int i = 0; i < 8; i++) { Serial.print(adc[i] > LINE_THRESHOLD_REAR[i] ? F("1") : F("0")); if (i < 7) Serial.print(F("     ")); }
  Serial.println();
  
  Serial.print(F("  WZone :  "));
  for (int i = 0; i < 8; i++) { Serial.print(adc[i] <= WHITE_ZONE_THRESHOLD_REAR[i] ? F("W") : F(".")); if (i < 7) Serial.print(F("     ")); }
  Serial.println();

  bool lost;
  float pos = lineGetPositionRear(lost);
  Serial.print(F("  Vi tri (quy ve he truoc): "));
  if (lost) Serial.println(F("MAT LINE"));
  else {
    Serial.print(pos, 3); Serial.print(F("  "));
    if      (abs(pos) < 0.3f) Serial.println(F("THANG"));
    else if (pos < 0)         Serial.println(F("lech TRAI"));
    else                      Serial.println(F("lech PHAI"));
  }
  int wz = countWhiteZoneRear();
  Serial.print(F("  WhiteZone (SAU): ")); Serial.print(wz); Serial.print(F("/")); Serial.print(WHITE_ZONE_MIN_COUNT);
  Serial.println(wz >= WHITE_ZONE_MIN_COUNT ? F(" -> DAO CHIEU!") : F(" (binh thuong)"));
  Serial.println(F("-----------------------------------------"));
}

void calibDoWhite() {
  Serial.println(F("[CALIB TRUOC] Dang do nen TRANG... (20 mau)"));
  for (int i = 0; i < 8; i++) calibWhite[i] = 0;
  for (int s = 0; s < 20; s++) { wdt_reset(); for (int i = 0; i < 8; i++) calibWhite[i] += analogRead(LINE_PINS[i]); delay(10); wdt_reset(); }
  Serial.print(F("[CALIB TRUOC] Nen TRANG: "));
  for (int i = 0; i < 8; i++) { calibWhite[i] /= 20; Serial.print(calibWhite[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println(); calibWhiteDone = true; calibBlackDone = false;
}

void calibDoBlack() {
  if (!calibWhiteDone) { Serial.println(F("[CALIB TRUOC] Goi 'calib' truoc!")); return; }
  Serial.println(F("[CALIB TRUOC] Dang do LINE DEN... (20 mau)"));
  for (int i = 0; i < 8; i++) calibBlack[i] = 0;
  for (int s = 0; s < 20; s++) { wdt_reset(); for (int i = 0; i < 8; i++) calibBlack[i] += analogRead(LINE_PINS[i]); delay(10); wdt_reset(); }
  Serial.print(F("[CALIB TRUOC] Line DEN: "));
  for (int i = 0; i < 8; i++) { calibBlack[i] /= 20; Serial.print(calibBlack[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println(); calibBlackDone = true;
}

void calibSave() {
  if (!calibWhiteDone || !calibBlackDone) { Serial.println(F("[CALIB TRUOC] Chua du du lieu (calib + calibline)")); return; }
  Serial.print(F("[CALIB TRUOC] Ap dung: "));
  for (int i = 0; i < 8; i++) { LINE_THRESHOLD[i] = (calibWhite[i]+calibBlack[i])/2; Serial.print(LINE_THRESHOLD[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
}

void calibShow() {
  Serial.println(F("[CALIB TRUOC] Nguong hien tai:"));
  Serial.print(F("  LINE_THRESHOLD: "));
  for (int i = 0; i < 8; i++) { Serial.print(LINE_THRESHOLD[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
}

void calibDoWhiteRear() {
  Serial.println(F("[CALIB SAU] Dang do nen TRANG... (20 mau)"));
  for (int i = 0; i < 8; i++) calibWhiteRear[i] = 0;
  for (int s = 0; s < 20; s++) { wdt_reset(); for (int i = 0; i < 8; i++) calibWhiteRear[i] += analogRead(LINE_PINS_REAR[i]); delay(10); wdt_reset(); }
  Serial.print(F("[CALIB SAU] Nen TRANG: "));
  for (int i = 0; i < 8; i++) { calibWhiteRear[i] /= 20; Serial.print(calibWhiteRear[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println(); calibWhiteRearDone = true; calibBlackRearDone = false;
}

void calibDoBlackRear() {
  if (!calibWhiteRearDone) { Serial.println(F("[CALIB SAU] Goi 'calibr' truoc!")); return; }
  Serial.println(F("[CALIB SAU] Dang do LINE DEN... (20 mau)"));
  for (int i = 0; i < 8; i++) calibBlackRear[i] = 0;
  for (int s = 0; s < 20; s++) { wdt_reset(); for (int i = 0; i < 8; i++) calibBlackRear[i] += analogRead(LINE_PINS_REAR[i]); delay(10); wdt_reset(); }
  Serial.print(F("[CALIB SAU] Line DEN: "));
  for (int i = 0; i < 8; i++) { calibBlackRear[i] /= 20; Serial.print(calibBlackRear[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println(); calibBlackRearDone = true;
}

void calibSaveRear() {
  if (!calibWhiteRearDone || !calibBlackRearDone) { Serial.println(F("[CALIB SAU] Chua du du lieu (calibr + calibrline)")); return; }
  Serial.print(F("[CALIB SAU] Ap dung: "));
  for (int i = 0; i < 8; i++) { LINE_THRESHOLD_REAR[i] = (calibWhiteRear[i]+calibBlackRear[i])/2; Serial.print(LINE_THRESHOLD_REAR[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
}

void calibShowRear() {
  Serial.println(F("[CALIB SAU] Nguong hien tai:"));
  Serial.print(F("  LINE_THRESHOLD_REAR: "));
  for (int i = 0; i < 8; i++) { Serial.print(LINE_THRESHOLD_REAR[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
}

void calibDoWhiteZone() {
  Serial.println(F("[CALIBW TRUOC] Dat robot len DAI TRANG DAO CHIEU roi gõ calibwsave"));
  Serial.println(F("[CALIBW TRUOC] Dang do... (20 mau)"));
  for (int i = 0; i < 8; i++) calibWhiteZone[i] = 0;
  for (int s = 0; s < 20; s++) {
    wdt_reset();
    for (int i = 0; i < 8; i++) calibWhiteZone[i] += analogRead(LINE_PINS[i]);
    delay(10); wdt_reset();
  }
  Serial.print(F("[CALIBW TRUOC] ADC trung binh tren dai trang dac biet: "));
  for (int i = 0; i < 8; i++) { 
    calibWhiteZone[i] /= 20; 
    Serial.print(calibWhiteZone[i]); if (i<7) Serial.print(F(" ")); 
  }
  Serial.println();
  Serial.print(F("[CALIBW TRUOC] Nguong se ap dung (ADC + ")); Serial.print(WHITE_ZONE_MARGIN); 
  Serial.print(F("): "));
  for (int i = 0; i < 8; i++) {
    int th = min(1023, calibWhiteZone[i] + WHITE_ZONE_MARGIN);
    Serial.print(th); if (i<7) Serial.print(F(" "));
  }
  Serial.println();
  Serial.println(F("[CALIBW TRUOC] Goi 'calibwsave' de ap dung."));
  calibWhiteZoneDone = true;
}

void calibSaveWhiteZone() {
  if (!calibWhiteZoneDone) { Serial.println(F("[CALIBW TRUOC] Goi 'calibw' truoc!")); return; }
  Serial.print(F("[CALIBW TRUOC] Da ap dung WHITE_ZONE_THRESHOLD: "));
  for (int i = 0; i < 8; i++) {
    WHITE_ZONE_THRESHOLD[i] = min(1023, calibWhiteZone[i] + WHITE_ZONE_MARGIN);
    Serial.print(WHITE_ZONE_THRESHOLD[i]); if (i<7) Serial.print(F(" "));
  }
  Serial.println();
  Serial.println(F("[CALIBW TRUOC] OK — Robot se dao chieu khi >=7 sensor dat nguong nay."));
}

void calibShowWhiteZone() {
  Serial.println(F("[CALIBW TRUOC] WHITE_ZONE_THRESHOLD hien tai:"));
  Serial.print(F("  "));
  for (int i = 0; i < 8; i++) { Serial.print(WHITE_ZONE_THRESHOLD[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
  Serial.println(F("  (0 = chua calib, robot se KHONG dao chieu)"));
}

void calibDoWhiteZoneRear() {
  Serial.println(F("[CALIBW SAU] Dat robot len DAI TRANG DAO CHIEU (phai sau) roi gõ calibwrsave"));
  Serial.println(F("[CALIBW SAU] Dang do... (20 mau)"));
  for (int i = 0; i < 8; i++) calibWhiteZoneRear[i] = 0;
  for (int s = 0; s < 20; s++) {
    wdt_reset();
    for (int i = 0; i < 8; i++) calibWhiteZoneRear[i] += analogRead(LINE_PINS_REAR[i]);
    delay(10); wdt_reset();
  }
  Serial.print(F("[CALIBW SAU] ADC trung binh: "));
  for (int i = 0; i < 8; i++) { 
    calibWhiteZoneRear[i] /= 20; 
    Serial.print(calibWhiteZoneRear[i]); if (i<7) Serial.print(F(" ")); 
  }
  Serial.println();
  Serial.print(F("[CALIBW SAU] Nguong se ap dung (ADC + ")); Serial.print(WHITE_ZONE_MARGIN); 
  Serial.print(F("): "));
  for (int i = 0; i < 8; i++) {
    int th = min(1023, calibWhiteZoneRear[i] + WHITE_ZONE_MARGIN);
    Serial.print(th); if (i<7) Serial.print(F(" "));
  }
  Serial.println();
  Serial.println(F("[CALIBW SAU] Goi 'calibwrsave' de ap dung."));
  calibWhiteZoneRearDone = true;
}

void calibSaveWhiteZoneRear() {
  if (!calibWhiteZoneRearDone) { Serial.println(F("[CALIBW SAU] Goi 'calibwr' truoc!")); return; }
  Serial.print(F("[CALIBW SAU] Da ap dung WHITE_ZONE_THRESHOLD_REAR: "));
  for (int i = 0; i < 8; i++) {
    WHITE_ZONE_THRESHOLD_REAR[i] = min(1023, calibWhiteZoneRear[i] + WHITE_ZONE_MARGIN);
    Serial.print(WHITE_ZONE_THRESHOLD_REAR[i]); if (i<7) Serial.print(F(" "));
  }
  Serial.println();
  Serial.println(F("[CALIBW SAU] OK."));
}

void calibShowWhiteZoneRear() {
  Serial.println(F("[CALIBW SAU] WHITE_ZONE_THRESHOLD_REAR hien tai:"));
  Serial.print(F("  "));
  for (int i = 0; i < 8; i++) { Serial.print(WHITE_ZONE_THRESHOLD_REAR[i]); if (i<7) Serial.print(F(" ")); }
  Serial.println();
  Serial.println(F("  (0 = chua calib)"));
}

float safeReadFloat(const char* cmd) {
  while (odrv_ser.available()) odrv_ser.read();
  odrv_ser.print(cmd);
  wdt_reset();
  unsigned long t0 = millis();
  while (odrv_ser.available() == 0) {
    if (millis() - t0 > ODRIVE_READ_TIMEOUT_MS) { Serial.println(F("[ERR] ODrive timeout")); return NAN; }
    wdt_reset(); delay(5);
  }
  String resp = odrv_ser.readStringUntil('\n');
  resp.trim();
  if (resp.length() == 0) { Serial.println(F("[ERR] ODrive chuoi rong")); return NAN; }
  return resp.toFloat();
}

PS2X ps2x;
bool motorOn      = false;
bool ultraBlocked = false;
bool debugPS2     = false;
bool lineLoopMode = false;
unsigned long tLineLoop = 0;

enum AppMode { MODE_PS2, MODE_SER, MODE_LINE } mode = MODE_PS2;

float tgtL = 0, tgtR = 0;
bool  lineLost = false;

bool  lcdNeedsUpdate      = true;
bool  lcdLastMotorOn      = false;
bool  lcdLastUltraBlocked = false;
bool  lcdLastEstop        = false;
bool  lcdLastLineLost     = false;
AppMode lcdLastMode       = MODE_PS2;
LineDir lcdLastDir        = DIR_FORWARD;

unsigned long tWdt=0, tSend=0, tPrint=0, tUltra=0, tPS2Read=0;
unsigned long tCross=0, tR2=0, tTri=0, tSqr=0, tCircle=0;
bool wCross=false, wR2=false, wTri=false, wSqr=false, wCircle=false;

volatile bool estopFlag   = false;
volatile bool estopActive = false;

float currentSteer = 0.0f;

inline void updateRelay() { digitalWrite(RELAY_PIN, ultraBlocked ? HIGH : LOW); }

void sendVel(float l, float r) {
  odrive.SetVelocity(0, l);
  odrive.SetVelocity(1, r);
}

volatile unsigned long ultra1EchoStart = 0;
volatile unsigned long ultra1EchoDur   = 0;
volatile bool          ultra1EchoReady = false;

volatile unsigned long ultra2EchoStart = 0;
volatile unsigned long ultra2EchoDur   = 0;
volatile bool          ultra2EchoReady = false;

void ultra1EchoISR() {
  if (digitalRead(ULTRA1_ECHO)) ultra1EchoStart = micros();
  else { ultra1EchoDur = micros() - ultra1EchoStart; ultra1EchoReady = true; }
}

void ultra2EchoISR() {
  if (digitalRead(ULTRA2_ECHO)) ultra2EchoStart = micros();
  else { ultra2EchoDur = micros() - ultra2EchoStart; ultra2EchoReady = true; }
}

float getUltra1Cm() { if (!ultra1EchoReady) return 999.0f; ultra1EchoReady = false; return ultra1EchoDur * 0.017f; }
float getUltra2Cm() { if (!ultra2EchoReady) return 999.0f; ultra2EchoReady = false; return ultra2EchoDur * 0.017f; }

void triggerDualUltra() {
  digitalWrite(ULTRA1_TRIG, LOW); digitalWrite(ULTRA2_TRIG, LOW); delayMicroseconds(2);
  digitalWrite(ULTRA1_TRIG, HIGH); digitalWrite(ULTRA2_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(ULTRA1_TRIG, LOW); digitalWrite(ULTRA2_TRIG, LOW);
}

void updateLCD() {
  bool changed = (lcdLastMotorOn != motorOn) || (lcdLastUltraBlocked != ultraBlocked) ||
                 (lcdLastEstop != estopActive) || (lcdLastLineLost != lineLost) ||
                 (lcdLastMode != mode) || (lcdLastDir != lineDir) || lcdNeedsUpdate;
  if (!changed) return;
  if (millis() - tLcd < T_LCD) return;
  tLcd = millis();

  lcdLastMotorOn = motorOn; lcdLastUltraBlocked = ultraBlocked;
  lcdLastEstop = estopActive; lcdLastLineLost = lineLost;
  lcdLastMode = mode; lcdLastDir = lineDir; lcdNeedsUpdate = false;

  lcd.setCursor(0, 0);
  if (!motorOn) lcd.print(F("MOTOR OFF        "));
  else {
    switch (mode) {
      case MODE_PS2:  lcd.print(F("MODE: PS2        ")); break;
      case MODE_SER:  lcd.print(F("MODE: SERIAL    ")); break;
      case MODE_LINE:
        if        (lineLost)            lcd.print(F("MODE: LINE[LOST]"));
        else if (lineDir==DIR_REVERSE) lcd.print(F("MODE: LINE [LUI]"));
        else if (lineDir==DIR_BRAKING) lcd.print(F("MODE: LINE [BRK]"));
        else                           lcd.print(F("MODE: LINE      "));
        break;
    }
  }

  lcd.setCursor(0, 1);
  if (estopActive)                                      lcd.print(F("!! E-STOP !!    "));
  else if (ultraBlocked)                                lcd.print(F("!! VAT CAN !!   "));
  else if (mode == MODE_LINE && lineLost)               lcd.print(F("!! MAT LINE !!  "));
  else if (mode == MODE_LINE && lineDir==DIR_BRAKING) lcd.print(F("PHANH...        "));
  else if (mode == MODE_LINE && lineDir==DIR_REVERSE) {
    if      (abs(lineRearLastPos) < 0.3f) lcd.print(F("LUI - THANG     "));
    else if (lineRearLastPos < 0)         lcd.print(F("LUI - lech TRAI "));
    else                                  lcd.print(F("LUI - lech PHAI "));
  } else if (mode == MODE_LINE) {
    if      (abs(lineLastPos) < 0.3f) lcd.print(F("DI THANG        "));
    else if (lineLastPos < 0)         lcd.print(F("LECH TRAI->Trai "));
    else                              lcd.print(F("LECH PHAI->Phai "));
  } else lcd.print(F("                "));
}

void emergencyStop(const char* r) {
  Serial.print(F("[ESTOP] ")); Serial.println(r);
  sendVel(0, 0); tgtL = tgtR = 0;
  wdt_reset(); delay(80); wdt_reset();
  odrive.run_state(0, AXIS_STATE_IDLE, false); wdt_reset();
  odrive.run_state(1, AXIS_STATE_IDLE, false); wdt_reset();
  motorOn = false; ultraBlocked = false; lineLoopMode = false; estopActive = true;
  updateRelay(); lineResetPID(); lineLost = false;
  if (mode == MODE_LINE) { mode = MODE_PS2; Serial.println(F("[LINE] ESTOP -> PS2")); }
  lcdNeedsUpdate = true;
}

void softBrake(const char* r) { tgtL = tgtR = 0; Serial.print(F("[BRAKE] ")); Serial.println(r); }

void enableMotors() {
  odrv_ser << "sc\n"; wdt_reset(); delay(50); wdt_reset();
  bool ok0 = odrive.run_state(0, AXIS_STATE_CLOSED_LOOP_CONTROL, false); wdt_reset();
  bool ok1 = odrive.run_state(1, AXIS_STATE_CLOSED_LOOP_CONTROL, false); wdt_reset();
  if (ok0 && ok1) { motorOn = true; ultraBlocked = false; estopActive = false; updateRelay(); Serial.println(F("[OK] Motor ON")); }
  else { Serial.println(F("[ERR] Enable FAIL - goi 's'")); }
  lcdNeedsUpdate = true;
}

void disableMotors() {
  sendVel(0, 0); tgtL = tgtR = 0; wdt_reset(); delay(80); wdt_reset();
  odrive.run_state(0, AXIS_STATE_IDLE, false); wdt_reset();
  odrive.run_state(1, AXIS_STATE_IDLE, false); wdt_reset();
  motorOn = false; estopActive = false; ultraBlocked = false; lineLoopMode = false;
  updateRelay(); lineResetPID(); lineLost = false;
  if (mode == MODE_LINE) { mode = MODE_PS2; }
  lcdNeedsUpdate = true; Serial.println(F("[OK] Motor OFF"));
}

void checkEmergency() {
  if (estopFlag) { estopFlag = false; emergencyStop("PS2/Serial"); return; }
  if (millis() - tUltra < T_ULTRA) return;
  tUltra = millis();
  triggerDualUltra();
  float d1 = getUltra1Cm(); float d2 = getUltra2Cm();
  float d = min(d1, d2);

  static uint8_t       blockCnt  = 0;
  static uint8_t       clearCnt  = 0;
  static unsigned long relayOnAt = 0;

  if (!ultraBlocked) {
    if (d < SAFE_CM) {
      if (++blockCnt >= BLOCK_CONFIRM) {
        blockCnt = 0; ultraBlocked = true; relayOnAt = millis(); updateRelay();
        Serial.print(F("[ULTRA] ")); Serial.print(d); Serial.println(F("cm BLOCK"));
        if (mode == MODE_LINE) softBrake("Ultra-LINE");
        lcdNeedsUpdate = true;
      }
    } else blockCnt = 0;
  } else {
    if (d > CLEAR_CM) {
      if (++clearCnt >= CLEAR_CONFIRM && millis() - relayOnAt >= RELAY_HOLD_MS) {
        clearCnt = 0; ultraBlocked = false; updateRelay();
        Serial.println(F("[ULTRA] OK + Relay OFF"));
        lcdNeedsUpdate = true;
      }
    } else if (d < SAFE_CM) clearCnt = 0;
  }
}

float joyAxis(byte raw) {
  int c = (int)raw - 128;
  if (abs(c) < DEADZONE) return 0.0f;
  float n = constrain((float)(abs(c) - DEADZONE) / (128.0f - DEADZONE), 0.0f, 1.0f);
  return (c > 0 ? 1.0f : -1.0f) * n;
}

void handlePS2() {
  if (millis() - tPS2Read < T_PS2_READ) return;
  tPS2Read = millis();

  bool ok = ps2x.read_gamepad(false, 0);
  byte ly = ps2x.Analog(PSS_LY);
  byte rx = ps2x.Analog(PSS_RX);

  if (debugPS2) {
    Serial.print(F("[PS2] SPI=")); Serial.print(ok);
    Serial.print(F(" LY=")); Serial.print(ly);
    Serial.print(F(" RX=")); Serial.println(rx);
  }

  if (!ok) { currentSteer = 0; return; }

  bool cn = ps2x.Button(PSB_CROSS);
  if (cn && !wCross && millis() - tCross >= T_DEBOUNCE) { tCross = millis(); wCross = true; estopFlag = true; return; }
  if (!cn) wCross = false;

  bool r2 = ps2x.Button(PSB_R2);
  if (r2 && !wR2 && millis() - tR2 >= T_DEBOUNCE) { tR2 = millis(); wR2 = true; if (!motorOn) enableMotors(); else disableMotors(); return; }
  if (!r2) wR2 = false;

  bool ci = ps2x.Button(PSB_CIRCLE);
  if (ci && !wCircle && millis() - tCircle >= T_DEBOUNCE) {
    tCircle = millis(); wCircle = true;
    if (mode != MODE_LINE) {
      if (!motorOn) Serial.println(F("[LINE] Motor chua bat"));
      else { mode = MODE_LINE; lineLost = false; lineResetPID(); softBrake("->LINE"); lcdNeedsUpdate = true; }
    } else { mode = MODE_PS2; lineLost = false; lineResetPID(); softBrake("->PS2"); lcdNeedsUpdate = true; }
  }
  if (!ci) wCircle = false;

  if (ps2x.Button(PSB_L1)) {
    if (mode == MODE_LINE) { mode = MODE_PS2; lineLost = false; lineResetPID(); lcdNeedsUpdate = true; }
    softBrake("L1"); return;
  }

  if (mode == MODE_LINE && abs(joyAxis(ly)) > 0.0f) { mode = MODE_PS2; lineLost = false; lineResetPID(); lcdNeedsUpdate = true; }

  {
    bool tri = ps2x.Button(PSB_TRIANGLE);
    if (tri && !wTri && millis() - tTri >= T_DEBOUNCE) {
      tTri = millis(); wTri = true;
      if (mode == MODE_LINE) { mode = MODE_PS2; lineLost = false; lineResetPID(); }
      if (mode != MODE_SER)  { mode = MODE_SER; softBrake("->SER"); lcdNeedsUpdate = true; }
    }
    if (!tri) wTri = false;

    bool sq = ps2x.Button(PSB_SQUARE);
    if (sq && !wSqr && millis() - tSqr >= T_DEBOUNCE) {
      tSqr = millis(); wSqr = true;
      if (mode == MODE_LINE) { mode = MODE_PS2; lineLost = false; lineResetPID(); }
      else if (mode != MODE_PS2) { mode = MODE_PS2; softBrake("->PS2"); }
      lcdNeedsUpdate = true;
    }
    if (!sq) wSqr = false;
  }

  if (mode == MODE_PS2) {
    float thrAxis  = joyAxis(ly);
    float steerAxis =  joyAxis(rx);
    currentSteer   = steerAxis;
    float baseVel  = thrAxis * FIXED_VEL;
    float steerVel = steerAxis * FIXED_VEL * STEER_RATIO;
    tgtL = constrain( baseVel - steerVel, -MAX_VEL, MAX_VEL);
    tgtR = constrain(-(baseVel + steerVel), -MAX_VEL, MAX_VEL);
  }
}

void handleSerial() {
  if (lineLoopMode) {
    if (Serial.available()) { Serial.read(); lineLoopMode = false; Serial.println(F("[LINE] Dung lineloop.")); return; }
    if (millis() - tLineLoop >= 200) {
      tLineLoop = millis(); wdt_reset();
      printLineSensor();
      printLineSensorRear();
    }
    return;
  }

  if (!Serial.available()) return;
  String s = Serial.readStringUntil('\n'); s.trim();
  if (!s.length()) return;

  if        (s == "e")                 { if (!motorOn) enableMotors(); else emergencyStop("Serial"); }
  else if (s == "d")                 { estopFlag = true; }
  else if (s == "stop" || s == "brake") { softBrake("Serial"); }
  else if (s == "s")                 { odrv_ser << "sc\n"; wdt_reset(); Serial.println(F("[OK] Cleared")); }
  else if (s == "b") { float v = safeReadFloat("r vbus_voltage\n"); if (!isnan(v)) { Serial.print(F("Vbus=")); Serial.print(v, 2); Serial.println(F("V")); } }
  else if (s == "vel") {
    float v0 = safeReadFloat("r axis0.encoder.vel_estimate\n"); wdt_reset();
    float v1 = safeReadFloat("r axis1.encoder.vel_estimate\n"); wdt_reset();
    if (!isnan(v0) && !isnan(v1)) { Serial.print(F("Axis0=")); Serial.print(v0,3); Serial.print(F("  Axis1=")); Serial.println(v1,3); }
  }
  else if (s == "raw") {
    Serial.println(F("[RAW] Gui 'r vbus_voltage'...")); while (odrv_ser.available()) odrv_ser.read();
    odrv_ser.print("r vbus_voltage\n"); wdt_reset(); delay(150); wdt_reset(); delay(150); wdt_reset();
    if (!odrv_ser.available()) Serial.println(F("[RAW] Khong co phan hoi!"));
    else { Serial.print(F("[RAW] '")); while (odrv_ser.available()) { char c = odrv_ser.read(); if (c=='\n') Serial.print(F("\\n")); else if (c=='\r') Serial.print(F("\\r")); else Serial.write(c); } Serial.println(F("'")); }
  }

  else if (s == "line")     { printLineSensor(); }
  else if (s == "liner")    { printLineSensorRear(); }
  else if (s == "lineloop") { lineLoopMode = true; tLineLoop = 0; Serial.println(F("[LINE] In lien tuc ca 2 dai. Phim bat ki de dung.")); }

  else if (s == "calib")     { calibDoWhite(); }
  else if (s == "calibline") { calibDoBlack(); }
  else if (s == "calibshow") { calibShow(); }
  else if (s == "calibsave") { calibSave(); }

  else if (s == "calibr")     { calibDoWhiteRear(); }
  else if (s == "calibrline") { calibDoBlackRear(); }
  else if (s == "calibrshow") { calibShowRear(); }
  else if (s == "calibrsave") { calibSaveRear(); }

  else if (s == "calibw")     { calibDoWhiteZone(); }
  else if (s == "calibwsave") { calibSaveWhiteZone(); }
  else if (s == "calibwshow") { calibShowWhiteZone(); }

  else if (s == "calibwr")     { calibDoWhiteZoneRear(); }
  else if (s == "calibwrsave") { calibSaveWhiteZoneRear(); }
  else if (s == "calibwrshow") { calibShowWhiteZoneRear(); }

  else if (s.startsWith("linespeed")) {
    String arg = s.substring(9); arg.trim();
    if (!arg.length()) { Serial.print(F("[LINE] VEL=")); Serial.print(LINE_MAX_VEL,2); Serial.print(F("  STEER_MAX=")); Serial.println(LINE_MAX_VEL*0.5f,3); }
    else { float v = arg.toFloat(); if (v<0.05f||v>MAX_VEL) Serial.println(F("[ERR] Phai trong khoang 0.05..3.0")); else { LINE_MAX_VEL=v; Serial.print(F("[LINE] VEL=")); Serial.print(LINE_MAX_VEL,2); Serial.print(F("  STEER_MAX=")); Serial.println(LINE_MAX_VEL*0.5f,3); } }
  }
  else if (s == "f") {
    Serial.print(F("[MODE] ")); Serial.println(mode==MODE_PS2?F("PS2"):mode==MODE_SER?F("SER"):F("LINE"));
    Serial.print(F("[MOTOR] ")); Serial.println(motorOn?F("ON"):F("OFF"));
    Serial.print(F("[ESTOP] ")); Serial.println(estopActive?F("ACTIVE"):F("clear"));
    Serial.print(F("[ULTRA] ")); Serial.println(ultraBlocked?F("BLOCKED"):F("clear"));
    Serial.print(F("[LINE] VEL=")); Serial.print(LINE_MAX_VEL,2); Serial.print(F("  STEER_MAX=")); Serial.println(LINE_MAX_STEER,3);
    Serial.print(F("[DIR] ")); Serial.println(lineDir==DIR_FORWARD?F("TIEN"):lineDir==DIR_BRAKING?F("PHANH"):F("LUI"));
    printLinePID();
    calibShow(); calibShowRear();
    calibShowWhiteZone(); calibShowWhiteZoneRear();
  }
  else if (s == "ps2")     { debugPS2 = !debugPS2; Serial.println(debugPS2?F("[DEBUG] PS2 ON"):F("[DEBUG] PS2 OFF")); }
  else if (s == "linepid") { printLinePID(); }
  else if (s.startsWith("lkp")) { lineKp = s.substring(3).toFloat(); Serial.print(F("[LINE] Kp=")); Serial.println(lineKp,3); }
  else if (s.startsWith("lki")) { lineKi = s.substring(3).toFloat(); lineIntegral=0; Serial.print(F("[LINE] Ki=")); Serial.println(lineKi,3); }
  else if (s.startsWith("lkd")) { lineKd = s.substring(3).toFloat(); Serial.print(F("[LINE] Kd=")); Serial.println(lineKd,3); }
  else if (s.startsWith("lka")) { lineSteerAlpha = constrain(s.substring(3).toFloat(),0.05f,1.0f); Serial.print(F("[LINE] Alpha=")); Serial.println(lineSteerAlpha,3); }
  else if (s.startsWith("lkr")) { lineSteerRateLimit = s.substring(3).toFloat(); if(lineSteerRateLimit<=0) lineSteerRateLimit=0.08f; Serial.print(F("[LINE] Rate Limit=")); Serial.println(lineSteerRateLimit,3); }

  else if (s == "tl" || s.startsWith("tl ")) {
    if (!motorOn) { Serial.println(F("[ERR] Motor off")); return; } if (mode==MODE_LINE) { Serial.println(F("[ERR] Dang MODE_LINE")); return; }
    String arg=s.substring(2); arg.trim(); float sv=(arg.length()>0)?constrain(arg.toFloat(),0.05f,MAX_VEL):FIXED_VEL*STEER_RATIO;
    tgtL=-sv; tgtR=-sv; Serial.print(F("[OK] XOAY TRAI L=")); Serial.print(tgtL,2); Serial.print(F(" R=")); Serial.println(tgtR,2);
  }
  else if (s == "tr" || s.startsWith("tr ")) {
    if (!motorOn) { Serial.println(F("[ERR] Motor off")); return; } if (mode==MODE_LINE) { Serial.println(F("[ERR] Dang MODE_LINE")); return; }
    String arg=s.substring(2); arg.trim(); float sv=(arg.length()>0)?constrain(arg.toFloat(),0.05f,MAX_VEL):FIXED_VEL*STEER_RATIO;
    tgtL=sv; tgtR=sv; Serial.print(F("[OK] XOAY PHAI L=")); Serial.print(tgtL,2); Serial.print(F(" R=")); Serial.println(tgtR,2);
  }
  else if (s.startsWith("v ")) {
    if (!motorOn) { Serial.println(F("[ERR] Motor off")); return; } if (mode==MODE_LINE) { Serial.println(F("[ERR] Dang MODE_LINE")); return; }
    String a=s.substring(2); a.trim(); int sp=a.indexOf(' ');
    float fwd = -constrain(sp>0?a.substring(0,sp).toFloat():a.toFloat(),-MAX_VEL,MAX_VEL);
    float str=sp>0?constrain(a.substring(sp+1).toFloat(),-MAX_VEL,MAX_VEL):0;
    tgtL=constrain(fwd+str,-MAX_VEL,MAX_VEL); tgtR=constrain(-(fwd-str),-MAX_VEL,MAX_VEL);
    Serial.print(F("[OK] L=")); Serial.print(tgtL,2); Serial.print(F(" R=")); Serial.println(tgtR,2);
  }
}

void updateMotorControl() {
  if (!motorOn) return;

  if (mode == MODE_LINE) {
    bool lost;

    if (lineDir == DIR_BRAKING) {
      tgtL = tgtR = 0;
      if (millis() - tBrakeStart >= REVERSE_BRAKE_MS) {
        if (dirBeforeBrake == DIR_FORWARD) {
          lineDir = DIR_REVERSE;
          Serial.println(F("[LINE] BAT DAU LUI"));
        } else {
          lineDir = DIR_FORWARD;
          Serial.println(F("[LINE] BAT DAU TIEN"));
        }
        linePrevPos = 0; lineIntegral = 0; lineLastPos = 0;
        lineSteerSmooth = 0; lineSteerPrev = 0; lineRearLastPos = 0;
        whiteZoneCount = 0; lineLost = false;
        lcdNeedsUpdate = true;
      }
      return;
    }

    else if (lineDir == DIR_FORWARD) {
      if (countWhiteZoneFront() >= WHITE_ZONE_MIN_COUNT) {
        if (++whiteZoneCount >= WHITE_ZONE_CONFIRM) {
          Serial.println(F("[LINE] DAI TRANG (TRUOC) -> PHANH -> LUI"));
          dirBeforeBrake = DIR_FORWARD;
          lineDir     = DIR_BRAKING;
          tBrakeStart = millis();
          tgtL = tgtR = 0;
          whiteZoneCount = 0;
          lcdLastDir    = DIR_FORWARD;
          lcdNeedsUpdate = true;
          return;
        }
      } else {
        whiteZoneCount = 0;
      }

      float pos    = lineGetPosition(lost);
      float pidOut = lineComputePID(pos);

      if (lost) {
        if (!lineLost) {
          lineLost = true;
          linePrevPos = 0; lineIntegral = 0;
          lineSteerSmooth = 0; lineSteerPrev = 0;
          tgtL = tgtR = 0; lcdNeedsUpdate = true;
          Serial.println(F("[LINE] MAT LINE (TIEN)"));
        }
        tgtL = tgtR = 0;
      } else {
        if (lineLost) { lineLost = false; lcdNeedsUpdate = true; }
        if (!ultraBlocked) {
          float lineVel   = LINE_MAX_VEL;
          float velScale  = lineVel / MAX_VEL;
          float scaledPid = pidOut * (0.5f + 0.5f * velScale);

          tgtL = constrain((-lineVel + scaledPid), -MAX_VEL, MAX_VEL);
          tgtR = constrain(( lineVel + scaledPid), -MAX_VEL, MAX_VEL);
        }
      }
    }

    else if (lineDir == DIR_REVERSE) {
      if (countWhiteZoneRear() >= WHITE_ZONE_MIN_COUNT) {
        if (++whiteZoneCount >= WHITE_ZONE_CONFIRM) {
          Serial.println(F("[LINE] DAI TRANG (SAU) -> PHANH -> TIEN"));
          dirBeforeBrake = DIR_REVERSE;
          lineDir     = DIR_BRAKING;
          tBrakeStart = millis();
          tgtL = tgtR = 0;
          whiteZoneCount = 0;
          lcdLastDir    = DIR_REVERSE;
          lcdNeedsUpdate = true;
          return;
        }
      } else {
        whiteZoneCount = 0;
      }

      float pos    = lineGetPositionRear(lost);
      float pidOut = lineComputePID(pos);

      if (lost) {
        if (!lineLost) {
          lineLost = true;
          linePrevPos = 0; lineIntegral = 0;
          lineSteerSmooth = 0; lineSteerPrev = 0;
          tgtL = tgtR = 0; lcdNeedsUpdate = true;
          Serial.println(F("[LINE] MAT LINE (LUI)"));
        }
        tgtL = tgtR = 0;
      } else {
        if (lineLost) { lineLost = false; lcdNeedsUpdate = true; }
        if (!ultraBlocked) {
          float lineVel   = LINE_MAX_VEL;
          float velScale  = lineVel / MAX_VEL;
          float scaledPid = pidOut * (0.5f + 0.5f * velScale);

          tgtL = constrain(( lineVel + scaledPid), -MAX_VEL, MAX_VEL);
          tgtR = constrain((-lineVel + scaledPid), -MAX_VEL, MAX_VEL);
        }
      }
    }
  }

  float sendL = tgtL; float sendR = tgtR;
  if (ultraBlocked) { if (sendL < 0) sendL = 0; if (sendR > 0) sendR = 0; }

  bool turnL = false, turnR = false;
  if (motorOn && !estopActive && !ultraBlocked) {
    if (mode == MODE_LINE && !lineLost) {
      if (lineDir == DIR_FORWARD) {
        if       (lineLastPos < -0.3f) turnL = true;
        else if (lineLastPos >  0.3f) turnR = true;
      } else if (lineDir == DIR_REVERSE) {
        if       (lineRearLastPos < -0.3f) turnR = true;
        else if (lineRearLastPos >  0.3f) turnL = true;
      }
    } else if (mode == MODE_PS2) {
      if       (currentSteer < -0.2f) turnL = true;
      else if (currentSteer >  0.2f) turnR = true;
    } else if (mode == MODE_SER) {
      if       (tgtL < -0.05f && tgtR < -0.05f) turnL = true;
      else if (tgtL >  0.05f && tgtR >  0.05f) turnR = true;
    }
  }
  digitalWrite(RELAY_TURN_LEFT,  turnL ? TURN_RELAY_ON : TURN_RELAY_OFF);
  digitalWrite(RELAY_TURN_RIGHT, turnR ? TURN_RELAY_ON : TURN_RELAY_OFF);

  if (millis() - tSend >= T_SEND) { tSend=millis(); sendVel(sendL, sendR); }

  static float prvL=0, prvR=0;
  bool justStopped = (abs(prvL)>0.05f||abs(prvR)>0.05f)&&(abs(sendL)<0.05f&&abs(sendR)<0.05f);
  bool changed     = abs(sendL-prvL)>0.05f||abs(sendR-prvR)>0.05f;
  bool periodic    = (millis()-tPrint>=T_PRINT)&&(abs(sendL)>0.05f||abs(sendR)>0.05f);

  if (justStopped || changed || periodic) {
    tPrint=millis(); prvL=sendL; prvR=sendR;
    if (mode == MODE_LINE) {
      const char* dirStr = (lineDir==DIR_FORWARD) ? "TIEN" : (lineDir==DIR_REVERSE) ? "LUI" : "BRK";
      Serial.print(F("[LINE/")); Serial.print(dirStr); Serial.print(F("] "));
      if (lineDir == DIR_FORWARD) {
        bool lost2; float p2 = lineGetPosition(lost2);
        Serial.print(F("pos=")); Serial.print(p2,2);
        if (!lost2) { if(abs(p2)<0.3f) Serial.print(F("(TG)")); else if(p2<0) Serial.print(F("(TR)")); else Serial.print(F("(PH)")); }
        if (lost2) Serial.print(F(" [LOST]"));
      } else if (lineDir == DIR_REVERSE) {
        bool lost2; float p2 = lineGetPositionRear(lost2);
        Serial.print(F("pos_r=")); Serial.print(p2,2);
        if (!lost2) { if(abs(p2)<0.3f) Serial.print(F("(TG)")); else if(p2<0) Serial.print(F("(TR)")); else Serial.print(F("(PH)")); }
        if (lost2) Serial.print(F(" [LOST]"));
      }
    } else { Serial.print(mode==MODE_PS2?F("[PS2]"):F("[SER]")); }
    Serial.print(F(" L=")); Serial.print(sendL,2); Serial.print(F(" R=")); Serial.print(sendR,2);
    if (ultraBlocked) Serial.println(F(" [BLK]")); else Serial.println();
  }
}

void setup() {
  wdt_disable();
  odrv_ser.begin(19200);
  Serial.begin(115200);
  while (!Serial);

  pinMode(ULTRA1_TRIG, OUTPUT); pinMode(ULTRA1_ECHO, INPUT);
  attachInterrupt(digitalPinToInterrupt(ULTRA1_ECHO), ultra1EchoISR, CHANGE);
  pinMode(ULTRA2_TRIG, OUTPUT); pinMode(ULTRA2_ECHO, INPUT);
  attachInterrupt(digitalPinToInterrupt(ULTRA2_ECHO), ultra2EchoISR, CHANGE);

  pinMode(RELAY_PIN, OUTPUT); digitalWrite(RELAY_PIN, LOW);
  pinMode(RELAY_TURN_LEFT, OUTPUT);  digitalWrite(RELAY_TURN_LEFT,  TURN_RELAY_OFF);
  pinMode(RELAY_TURN_RIGHT, OUTPUT); digitalWrite(RELAY_TURN_RIGHT, TURN_RELAY_OFF);

  odrv_ser << "sc\n"; wdt_reset(); delay(200); wdt_reset();
  odrv_ser << "w axis0.controller.config.vel_limit " << MAX_VEL << '\n';
  odrv_ser << "w axis0.motor.config.current_lim 4.0\n";
  odrv_ser << "w axis1.controller.config.vel_limit " << MAX_VEL << '\n';
  odrv_ser << "w axis1.motor.config.current_lim 4.0\n";
  wdt_reset(); delay(200); wdt_reset();

  Wire.begin();
  lcd.init(); lcd.backlight();
  lcd.setCursor(0,0); lcd.print(F("AGV Apbee        "));
  lcd.setCursor(0,1); lcd.print(F("Khoi dong...    "));

  ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_SEL, PS2_DAT, false, false);

  Serial.println(F("------------------------------------------------------------"));
  Serial.println(F("  Calib line TRUOC  : calib -> calibline -> calibsave"));
  Serial.println(F("  Calib line SAU    : calibr -> calibrline -> calibrsave"));
  Serial.println(F("  Calib DAI TRANG dao chieu (TRUOC): calibw -> calibwsave"));
  Serial.println(F("  Calib DAI TRANG dao chieu (SAU)  : calibwr -> calibwrsave"));
  Serial.println(F("  Xem cam bien      : line (truoc) | liner (sau)"));
  Serial.println(F("  Xem nguong WZone  : calibwshow | calibwrshow"));
  Serial.println(F("------------------------------------------------------------"));
  delay(1000);
  lcd.clear();
  lcdNeedsUpdate = true;
  wdt_enable(WDTO_500MS);

  Serial.begin(9600); 
  
  hori_serv.attach(servoPin1); 
  verti_serv.attach(servoPin2);
  
  hori_serv.write(90);
  verti_serv.write(90);
}

void loop() {
  if (millis() - tWdt >= T_WDT) { wdt_reset(); tWdt = millis(); }
  checkEmergency();
  handlePS2();
  handleSerial();
  updateMotorControl(); 
  updateLCD();
  // 1. LISTEN FOR PYTHON COMMANDS
  if (Serial.available() > 0) {
    char command = Serial.read();

    // --- PAN (Horizontal) ---
    if (command == 'L') {
      hori_serv.write(pan_left_speed);
      last_pan_time = millis(); // Reset the timer!
      is_panning = true;
    } else if (command == 'R') {
      hori_serv.write(pan_right_speed);
      last_pan_time = millis(); // Reset the timer!
      is_panning = true;
    } else if (command == 'X') {
      hori_serv.write(90); 
      is_panning = false;
    }
    
    // --- TILT (Vertical) ---
    if (command == 'U') {
      verti_serv.write(tilt_up_speed);
      last_tilt_time = millis(); // Reset the timer!
      is_tilting = true;
    } else if (command == 'D') {
      verti_serv.write(tilt_down_speed);
      last_tilt_time = millis(); // Reset the timer!
      is_tilting = true;
    } else if (command == 'Y') {
      verti_serv.write(90); 
      is_tilting = false;
    }
    
    // --- HOME ---
    if (command == 'H') {
      hori_serv.write(90);
      verti_serv.write(90);
      is_panning = false;
      is_tilting = false;
    }
  }

  // 2. THE WATCHDOG: AUTO-STOP IF PYTHON LAGS
  unsigned long current_time = millis();

  // If the servo is moving, but the timer has run out, force it to stop.
  if (is_panning && (current_time - last_pan_time > burst_duration)) {
    hori_serv.write(90);
    is_panning = false;
  }

  if (is_tilting && (current_time - last_tilt_time > burst_duration)) {
    verti_serv.write(90);
    is_tilting = false;
  }
}