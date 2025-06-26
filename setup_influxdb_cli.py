#!/usr/bin/env python3
"""
InfluxDB CLI Setup for TradeKnowledge
Automates the initial setup using InfluxDB CLI
"""

import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and return the result"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} - Success")
            return result.stdout.strip()
        else:
            logger.error(f"❌ {description} - Failed")
            logger.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"❌ {description} - Exception: {e}")
        return None

def setup_influxdb():
    """Setup InfluxDB using CLI"""
    
    # Configuration
    username = "admin"
    password = "tradeknowledge_admin_2024"
    org = "tradeknowledge"
    bucket = "market_data"
    
    logger.info("🚀 Setting up InfluxDB for TradeKnowledge")
    
    # Setup initial admin user and organization
    setup_cmd = f"""influx setup \
        --username {username} \
        --password {password} \
        --org {org} \
        --bucket {bucket} \
        --force"""
    
    result = run_command(setup_cmd, "Initial InfluxDB setup")
    
    if result:
        logger.info("📋 InfluxDB Setup Complete!")
        logger.info(f"Organization: {org}")
        logger.info(f"Bucket: {bucket}")
        logger.info(f"Username: {username}")
        
        # Get the token
        token_cmd = f"influx auth list --user {username} --hide-headers | head -1 | awk '{{print $3}}'"
        token = run_command(token_cmd, "Getting auth token")
        
        if token:
            logger.info(f"🔑 Token: {token}")
            logger.info("\n📝 Update your configuration:")
            logger.info(f"INFLUX_TOKEN={token}")
            
            # Update the database.py file
            try:
                with open('src/core/database.py', 'r') as f:
                    content = f.read()
                
                updated_content = content.replace(
                    'self.influx_token = "blackboard-super-secret-auth-token"',
                    f'self.influx_token = "{token}"'
                )
                
                with open('src/core/database.py', 'w') as f:
                    f.write(updated_content)
                
                logger.info("✅ Updated database.py with new token")
                
            except Exception as e:
                logger.error(f"Error updating database.py: {e}")
        
        return token
    else:
        logger.error("Setup failed")
        return None

if __name__ == "__main__":
    token = setup_influxdb()
    if token:
        print(f"\n🎉 InfluxDB setup complete! Token: {token}")
        sys.exit(0)
    else:
        print("\n❌ Setup failed")
        sys.exit(1)