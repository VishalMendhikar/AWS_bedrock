import boto3
import botocore.config
import json
import logging
from datetime import datetime
from uuid import uuid4

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def blog_generate_using_bedrock(blog_topic: str) -> str:
    prompt = f"""<s>[INST]Human: Write a 200 words blog on the topic {blog_topic} Assistant:[/INST]"""

    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="ap-south-1",
            config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 5, 'mode': 'adaptive'})
        )
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama3-8b-instruct-v1:0")
        response_content = response.get('body').read()
        response_data = json.loads(response_content)

        logger.info(f"Bedrock response: {response_data}")
        blog_details = response_data.get('generation', 'No content generated')
        return blog_details
    except Exception as e:
        logger.error(f"Error generating the blog: {e}")
        return ""

def save_blog_details_s3(s3_key: str, s3_bucket: str, generate_blog: str):
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generate_blog)
        logger.info("Blog saved to S3 successfully")
    except Exception as e:
        logger.error(f"Error saving blog to S3: {e}")

def lambda_handler(event, context):
    try:
        event_body = json.loads(event.get('body', '{}'))
        blog_topic = event_body.get('blog_topic')
        if not blog_topic:
            raise ValueError("Missing 'blog_topic' in the request body")

        generated_blog = blog_generate_using_bedrock(blog_topic=blog_topic)

        if generated_blog:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"blog-output/{current_time}_{uuid4().hex}.txt"
            s3_bucket = 'aws.bedrockcourse'
            save_blog_details_s3(s3_key, s3_bucket, generated_blog)
        else:
            logger.error("No blog was generated")

        return {
            'statusCode': 200,
            'body': json.dumps({'blog': generated_blog})
        }
    except Exception as e:
        logger.error(f"Error in Lambda handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
