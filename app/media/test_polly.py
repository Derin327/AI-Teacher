import boto3

polly = boto3.client("polly", region_name="ap-south-1")

response = polly.synthesize_speech(
    Text="Hello. Amazon Polly is connected successfully.",
    OutputFormat="mp3",
    VoiceId="Joanna",
    Engine="neural"
)

with open("test_polly.mp3", "wb") as f:
    f.write(response["AudioStream"].read())

print("SUCCESS")