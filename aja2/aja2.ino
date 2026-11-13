#include <Arduino.h>

// ===== 実験の設定 =====================================================
// 1: 同じ読み取り値に「圧縮なし」と「CSR」の両方の推論をかけ，PPM を2枚出力する
//    （2枚の MSE の差が，測定誤差を含まない「圧縮だけの影響」になる）
//    圧縮なしの重みは SRAM に載るため，H=60 程度までしか使えない
// 0: CSR のみ（H=100, 200 など大きい次元用）
#define COMPARE_DENSE 0

// 読み取りの安定化
#define SETTLE_DELAY 300   // ボタン押下後，紙と手の振動が収まるまで待つ時間 [ms]
#define READ_REPEAT  5     // 1画素あたりの読み取り回数（平均をとる）

// 2つのヘッダは配列名（bias_1 など）が重なるため，namespace で分けて読み込む
// どちらも同じ学習結果から出力したものを使うこと
namespace csr {
#include "model_parameters_csr.h"
}
#if COMPARE_DENSE
namespace dense {
#include "model_parameters.h"
}
#endif

#define INPUT_SIZE 3
#define HIDDEN_SIZE1 200            // ← 学習時の次元に合わせる
#define HIDDEN_SIZE2 200
#define OUTPUT_SIZE 3

// RGB点灯遅延
#define RgbFlashDelay 100

// ピン割り当て
#define PIN_IN A0
#define led_r_pin 8
#define led_g_pin 6
#define led_b_pin 7
#define buttonInputPin 3
#define buttonMinMaxPin 2

// ボタン押下を検知する変数 (余計な最適化を防ぐため、volatile宣言をしている)
volatile bool buttonInputPressed = false;
volatile bool buttonMinMaxPressed = false;

// RGB読み取り用変数
float r_max=512, g_max=512, b_max=512;
float r_min=0, g_min=0, b_min=0;

float rgb[3]{0, 0, 0};       // 1回分の生の読み取り値
float rgbAvg[3]{0, 0, 0};    // READ_REPEAT 回の平均
float rgbRange[3]{0, 0, 0};  // READ_REPEAT 回の最大−最小（読み取りのばらつきの目安）
float RGBInput[3]{0, 0, 0};
float RGBOutputCsr[3]{0, 0, 0};
float RGBOutputDense[3]{0, 0, 0};

// RGBデータメモリ
int N, M;
float **arrayCsr;
#if COMPARE_DENSE
float **arrayDense;
#endif
int n=0, m=0;

int pushed = 0;

// predict() の実行時間計測用 [us]
struct TimeStat {
    unsigned long last = 0;
    unsigned long sum = 0;
    unsigned long min = 0xFFFFFFFF;
    unsigned long max = 0;
    int count = 0;
    void add(unsigned long t) {
        last = t; sum += t; count++;
        if (t < min) min = t;
        if (t > max) max = t;
    }
    String summary() const {
        return "平均: " + String((float)sum / count, 2) + " us"
             + "  最小: " + String(min) + " us"
             + "  最大: " + String(max) + " us"
             + "  (n=" + String(count) + ")";
    }
};
TimeStat timeCsr;
#if COMPARE_DENSE
TimeStat timeDense;
#endif

float relu(float x) {
    return (x > 0) ? x : 0;
}

void buttonInput() {
    buttonInputPressed = true;
}

void buttonMinMax() {
    buttonMinMaxPressed = true;
}

float **allocateArray(int N, int M) {
    float **array = (float **)malloc(N * sizeof(float *)); // 行ポインタの確保
    if (array == NULL) {
        Serial.println("メモリ確保失敗");
        return NULL;
    }

    for (int i = 0; i < N; i++) {
        array[i] = (float *)malloc(M * sizeof(float)); // 各行のメモリ確保
        if (array[i] == NULL) {
            Serial.println("メモリ確保失敗（部分的）");
            // 確保済みのメモリを解放して終了
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
        for (int j = 0; j < M; j++) array[i][j] = 0;
    }
    return array;
}

void read(){
    for(int count = 0 ; count < 3 ; count++){
        if(count == 0){
            analogWrite(led_r_pin, 255);
            analogWrite(led_g_pin,   0);
            analogWrite(led_b_pin,   0);
            delay(RgbFlashDelay);
            rgb[0] = analogRead(PIN_IN);
        }else if(count == 1){
            analogWrite(led_r_pin,   0);
            analogWrite(led_g_pin, 255);
            analogWrite(led_b_pin,   0);
            delay(RgbFlashDelay);
            rgb[1] = analogRead(PIN_IN);
        }else{
            analogWrite(led_r_pin,   0);
            analogWrite(led_g_pin,   0);
            analogWrite(led_b_pin, 255);
            delay(RgbFlashDelay);
            rgb[2] = analogRead(PIN_IN);
        }
    }
    analogWrite(led_r_pin, 0);
    analogWrite(led_g_pin, 0);
    analogWrite(led_b_pin, 0);
}

// ボタンを押した「後」に READ_REPEAT 回読み，平均を rgbAvg に入れる
// （以前は loop() で常時読み続けた直前の値を使っていたため，紙をずらしている
//   最中の値が入ることがあり，MSE が大きくばらつく原因になっていた）
void readAveraged() {
    delay(SETTLE_DELAY);
    float sum[3] = {0, 0, 0};
    float lo[3] = {1e9, 1e9, 1e9};
    float hi[3] = {-1e9, -1e9, -1e9};
    for (int k = 0; k < READ_REPEAT; k++) {
        read();
        for (int c = 0; c < 3; c++) {
            sum[c] += rgb[c];
            if (rgb[c] < lo[c]) lo[c] = rgb[c];
            if (rgb[c] > hi[c]) hi[c] = rgb[c];
        }
    }
    for (int c = 0; c < 3; c++) {
        rgbAvg[c] = sum[c] / READ_REPEAT;
        rgbRange[c] = hi[c] - lo[c];
    }
}

// CSR形式の全結合層: y = W x + bias
void csr_layer(const float *val, const uint8_t *col, const uint16_t *row_ptr,
               const float *bias, const float *x, float *y, int rows) {
    for (int i = 0; i < rows; i++) {
        float sum = bias[i];
        for (uint16_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            sum += val[k] * x[col[k]];
        }
        y[i] = sum;
    }
}

void predictCsr(const float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
    float layer1[HIDDEN_SIZE1] = {0};
    float layer2[HIDDEN_SIZE2] = {0};
    // Layer 1
    csr_layer(csr::w1_val, csr::w1_col, csr::w1_ptr, csr::bias_1, input, layer1, HIDDEN_SIZE1);
    for (int i = 0; i < HIDDEN_SIZE1; i++) layer1[i] = relu(layer1[i]);
    // Layer 2
    csr_layer(csr::w2_val, csr::w2_col, csr::w2_ptr, csr::bias_2, layer1, layer2, HIDDEN_SIZE2);
    for (int i = 0; i < HIDDEN_SIZE2; i++) layer2[i] = relu(layer2[i]);
    // Layer 3（出力層，ReLUなし）
    csr_layer(csr::w3_val, csr::w3_col, csr::w3_ptr, csr::bias_3, layer2, output, OUTPUT_SIZE);
}

#if COMPARE_DENSE
// 圧縮なしの全結合層: y = W x + bias（W は rows×cols を行優先で格納）
void dense_layer(const float *W, const float *bias, const float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = bias[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

void predictDense(const float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
    float layer1[HIDDEN_SIZE1] = {0};
    float layer2[HIDDEN_SIZE2] = {0};
    dense_layer(dense::weight_1, dense::bias_1, input, layer1, HIDDEN_SIZE1, INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE1; i++) layer1[i] = relu(layer1[i]);
    dense_layer(dense::weight_2, dense::bias_2, layer1, layer2, HIDDEN_SIZE2, HIDDEN_SIZE1);
    for (int i = 0; i < HIDDEN_SIZE2; i++) layer2[i] = relu(layer2[i]);
    dense_layer(dense::weight_3, dense::bias_3, layer2, output, OUTPUT_SIZE, HIDDEN_SIZE2);
}
#endif

// 0-1 の出力を 0-255 にしてクリッピング
void toPixel(float out[3]) {
    for (int c = 0; c < 3; c++) out[c] = min(max(out[c] * 255, 0), 255);
}

// Arduino's analog input detects 0-5V with 10-bit values
// https://deviceplus.jp/arduino/arduino_f07/
void readAndProcess(){
    readAveraged();
    RGBInput[0] = (rgbAvg[0]-r_min)/r_max;
    RGBInput[1] = (rgbAvg[1]-g_min)/g_max;
    RGBInput[2] = (rgbAvg[2]-b_min)/b_max;

    Serial.println("Measured R, G, B: " + String(RGBInput[0]) + "," + String(RGBInput[1]) + "," + String(RGBInput[2])
        + "  (読み取りのばらつき R,G,B: " + String(rgbRange[0], 0) + "," + String(rgbRange[1], 0) + "," + String(rgbRange[2], 0) + ")");

    // ニューラルネットで読み込み値を修正
    // Serial 出力を計測区間に含めないよう，predict の直前・直後だけを測る
    // （unsigned の引き算なので micros() のオーバーフローをまたいでも正しい）
    unsigned long t0 = micros();
    predictCsr(RGBInput, RGBOutputCsr);
    timeCsr.add(micros() - t0);
    toPixel(RGBOutputCsr);
    Serial.println("Predicted(CSR)   R, G, B: " + String((int)RGBOutputCsr[0]) + "," + String((int)RGBOutputCsr[1]) + "," + String((int)RGBOutputCsr[2])
        + "  predict: " + String(timeCsr.last) + " us");

#if COMPARE_DENSE
    t0 = micros();
    predictDense(RGBInput, RGBOutputDense);
    timeDense.add(micros() - t0);
    toPixel(RGBOutputDense);
    Serial.println("Predicted(dense) R, G, B: " + String((int)RGBOutputDense[0]) + "," + String((int)RGBOutputDense[1]) + "," + String((int)RGBOutputDense[2])
        + "  predict: " + String(timeDense.last) + " us");
#endif
}

void printPPM(const char *label, float **array) {
    Serial.println(String("=== ") + label + " ===");
    Serial.println("--------------------------------------------------");
    Serial.println("P3");
    Serial.println(String((int)(M/3))+" "+String(N));
    Serial.println("255");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Serial.print((int)array[i][j]);
            Serial.print(" ");
        }
        Serial.println();
    }
    Serial.println("--------------------------------------------------");
    Serial.println();
}

void setup() {
    Serial.begin(9600);
    pinMode(led_r_pin, OUTPUT);
    pinMode(led_g_pin, OUTPUT);
    pinMode(led_b_pin, OUTPUT);
    analogWrite(led_r_pin, 0);
    analogWrite(led_b_pin, 0);
    analogWrite(led_g_pin, 0);

    pinMode(buttonInputPin, INPUT_PULLUP);
    pinMode(buttonMinMaxPin, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(buttonInputPin), buttonInput, FALLING);
    attachInterrupt(digitalPinToInterrupt(buttonMinMaxPin), buttonMinMax, FALLING);

    Serial.println(String("H=") + HIDDEN_SIZE1 + "  COMPARE_DENSE=" + COMPARE_DENSE
        + "  READ_REPEAT=" + READ_REPEAT + "  SETTLE_DELAY=" + SETTLE_DELAY + "ms");
    Serial.println("画像の Height（行数）を入力してください: ");
    while (Serial.available() == 0) {
        // 何もせずに待機
    }
    N = Serial.parseInt();
    M = N*3; // RGBの値を入れるために3倍しておく
    Serial.println("Height: " + String(N));
    Serial.println("Width: " + String((int)(M/3)));

    arrayCsr = allocateArray(N, M);
#if COMPARE_DENSE
    arrayDense = allocateArray(N, M);
    if (arrayDense == NULL) {
        Serial.println("配列の確保に失敗");
        return;
    }
#endif
    if (arrayCsr == NULL) {
        Serial.println("配列の確保に失敗");
        return;
    }
    // ボタン待ちの間に押されたフラグを捨てる
    buttonInputPressed = false;
    buttonMinMaxPressed = false;
}

void loop() {
    // 以前は常時読み取りを続けていたが，ボタンを押してから読むように変更した
    if (buttonInputPressed) {
        readAndProcess();
        for (int c = 0; c < 3; c++) {
            arrayCsr[n][m+c] = RGBOutputCsr[c];
#if COMPARE_DENSE
            arrayDense[n][m+c] = RGBOutputDense[c];
#endif
        }

        Serial.println(String(n)+" 行目 "+String((int)(m/3))+" 列目 の画素を読み込みました。");
        Serial.println();

        if((m+=3)>=M){
            m = 0 ;
            if((n+=1)>=N){
                n = 0;
                Serial.println("画像の読み取りが完了しました。");
                Serial.println("以下のテキストを画像ファイル（.ppm）に貼り付けてください。");
                printPPM("CSR", arrayCsr);
#if COMPARE_DENSE
                printPPM("圧縮なし（dense）", arrayDense);
#endif
                // PPM のコピー範囲に混ざらないよう，区切り線の後に出力する
                Serial.println("[predict 実行時間] H=" + String(HIDDEN_SIZE1));
                Serial.println("  CSR  : " + timeCsr.summary());
#if COMPARE_DENSE
                Serial.println("  dense: " + timeDense.summary());
#endif
                delay(500);
                NVIC_SystemReset();
            }
        }
        // 処理中（約 SETTLE_DELAY + READ_REPEAT×300ms）のチャタリングや二度押しは無視する
        buttonInputPressed = false;
    }
    if (buttonMinMaxPressed) {
        // キャリブレーションも，押した後に平均をとった値を使う
        readAveraged();
        if(pushed%2==0){
            r_min=rgbAvg[0], g_min=rgbAvg[1], b_min=rgbAvg[2];
            Serial.println("最小値が更新されました（R,G,B）："+String(r_min)+","+String(g_min)+","+String(b_min));
        }
        else{
            r_max=rgbAvg[0]-r_min, g_max=rgbAvg[1]-g_min, b_max=rgbAvg[2]-b_min;
            Serial.println("最大値が更新されました（R,G,B）："+String(r_max)+","+String(g_max)+","+String(b_max));
        }
        pushed++;
        buttonMinMaxPressed = false;  // フラグをリセット（処理中の二度押しは無視）
    }
}
