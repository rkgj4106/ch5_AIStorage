from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "documents" 

if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
client.fput_object(
    "documents",
    "report.pdf",
    "report.pdf"
)
