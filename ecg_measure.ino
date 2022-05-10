// My code

unsigned long previousMillis = 0;        // will store last time LED was updated
const long interval = 10;           // interval at which to blink (milliseconds)

uint8_t Lo1 = D0;
uint8_t Lo2 = D2;


void setup()
{
  Serial.begin(9600);
  pinMode(Lo2, INPUT); // Setup for leads off detection LO +
  pinMode(Lo1, INPUT); // Setup for leads off detection LO -
}

void loop()
{
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;
    Serial.println(analogRead(A0));
  }

}
