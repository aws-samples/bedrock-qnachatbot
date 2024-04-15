#!/bin/bash
# Updates the S3 Bucket name and the AWS Region name
# where the app is deployed

bucket_name=$1
aws_region_name=$2

sed -i "s/<S3_Bucket_Name>/${bucket_name}/g" /app/src/utils.py
sed -i "s/<AWS_Region>/${aws_region_name}/g" /app/src/utils.py
export AWS_DEFAULT_REGION=${aws_region_name}