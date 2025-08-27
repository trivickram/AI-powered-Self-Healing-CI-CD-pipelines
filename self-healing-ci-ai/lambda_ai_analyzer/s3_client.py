import boto3
import logging
from typing import Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class S3Client:
    """
    Client for handling S3 operations related to CI/CD logs.
    """
    
    def __init__(self, bucket_name: str, region: str):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
        logger.info(f"📦 S3 Client initialized - bucket: {bucket_name}, region: {region}")
    
    def get_log_file(self, log_key: str) -> Optional[str]:
        """
        Retrieve log file content from S3.
        
        Args:
            log_key: S3 key for the log file (e.g., 'logs/123456789.txt')
            
        Returns:
            Log file content as string, or None if not found
        """
        try:
            logger.info(f"📥 Retrieving log file: s3://{self.bucket_name}/{log_key}")
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=log_key
            )
            
            content = response['Body'].read().decode('utf-8')
            
            logger.info(f"✅ Successfully retrieved log file ({len(content)} characters)")
            return content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"❌ Log file not found: s3://{self.bucket_name}/{log_key}")
            else:
                logger.error(f"❌ S3 error retrieving log file: {e}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Unexpected error retrieving log file: {e}")
            return None
    
    def list_log_files(self, prefix: str = 'logs/', max_keys: int = 100) -> list:
        """
        List available log files in S3.
        
        Args:
            prefix: S3 prefix to filter log files
            max_keys: Maximum number of files to return
            
        Returns:
            List of log file keys
        """
        try:
            logger.info(f"📋 Listing log files with prefix: {prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            if 'Contents' not in response:
                logger.info("📭 No log files found")
                return []
            
            log_files = [obj['Key'] for obj in response['Contents']]
            logger.info(f"📄 Found {len(log_files)} log files")
            
            return log_files
            
        except Exception as e:
            logger.error(f"❌ Error listing log files: {e}")
            return []
    
    def upload_analysis_result(self, run_id: str, analysis: dict) -> bool:
        """
        Upload AI analysis result to S3 for future reference.
        
        Args:
            run_id: GitHub Actions run ID
            analysis: AI analysis result dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            from datetime import datetime
            
            analysis_key = f"analysis/{run_id}.json"
            
            # Add metadata
            analysis_with_metadata = {
                'run_id': run_id,
                'timestamp': datetime.utcnow().isoformat(),
                'analysis': analysis
            }
            
            logger.info(f"📤 Uploading analysis result: s3://{self.bucket_name}/{analysis_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=analysis_key,
                Body=json.dumps(analysis_with_metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info("✅ Analysis result uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading analysis result: {e}")
            return False
    
    def get_signed_url(self, log_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for accessing a log file.
        
        Args:
            log_key: S3 key for the log file
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Signed URL string, or None if error
        """
        try:
            logger.info(f"🔗 Generating signed URL for: {log_key}")
            
            signed_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': log_key},
                ExpiresIn=expiration
            )
            
            logger.info("✅ Signed URL generated successfully")
            return signed_url
            
        except Exception as e:
            logger.error(f"❌ Error generating signed URL: {e}")
            return None
    
    def cleanup_old_logs(self, days_old: int = 30) -> int:
        """
        Clean up log files older than specified days.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            logger.info(f"🧹 Cleaning up logs older than {days_old} days")
            
            # List all objects in the logs prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='logs/'
            )
            
            if 'Contents' not in response:
                logger.info("📭 No log files to clean up")
                return 0
            
            deleted_count = 0
            
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    try:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        deleted_count += 1
                        logger.info(f"🗑️ Deleted old log: {obj['Key']}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error deleting {obj['Key']}: {e}")
            
            logger.info(f"✅ Cleanup completed - deleted {deleted_count} files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return 0
