# TradeKnowledge API Deployment Guide - Phase 4

## Overview

TradeKnowledge Phase 4 introduces a production-ready REST API with authentication, user management, and comprehensive security features. This guide covers deployment options and configuration.

## 🚀 Quick Start

### 1. Initialize the API System

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize API and create admin user
python scripts/init_api.py

# Start the API server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📋 API Endpoints Overview

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration (if enabled)
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Search
- `POST /api/v1/search/query` - Main search endpoint
- `GET /api/v1/search/context/{chunk_id}` - Get chunk context
- `GET /api/v1/search/autocomplete` - Query autocomplete
- `GET /api/v1/search/similar/{result_id}` - Find similar content
- `POST /api/v1/search/feedback` - Submit search feedback
- `GET /api/v1/search/trending` - Get trending queries
- `GET /api/v1/search/export/{format}` - Export search results

### Books Management
- `GET /api/v1/books/list` - List all books
- `GET /api/v1/books/{book_id}` - Get book details
- `POST /api/v1/books/upload` - Upload new book
- `DELETE /api/v1/books/{book_id}` - Delete book
- `PUT /api/v1/books/{book_id}` - Update book metadata

### Analytics
- `GET /api/v1/analytics/search` - Search analytics
- `GET /api/v1/analytics/usage` - System usage metrics
- `GET /api/v1/analytics/performance` - Performance metrics

### Administration
- `GET /api/v1/admin/users` - List users
- `POST /api/v1/admin/users` - Create user
- `PUT /api/v1/admin/users/{user_id}` - Update user
- `DELETE /api/v1/admin/users/{user_id}` - Delete user
- `GET /api/v1/admin/system/status` - System status
- `POST /api/v1/admin/system/config` - Update configuration

## 🔐 Authentication & Authorization

### User Roles

- **admin**: Full system access
- **editor**: Read/write access, can upload books
- **user**: Read access only
- **viewer**: Read-only access

### JWT Token Authentication

```python
# Login to get token
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "admin",
    "password": "your_password"
})
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.post("http://localhost:8000/api/v1/search/query", 
                        headers=headers, json={"query": "algorithmic trading"})
```

## ⚙️ Configuration

### Environment Variables

```bash
# Security
JWT_SECRET_KEY=your-secret-key-here
ENABLE_REGISTRATION=false

# API Settings
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
RATE_LIMIT_PER_MINUTE=60
MAX_FILE_SIZE_MB=100
REQUEST_TIMEOUT=300

# Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379

# Embeddings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=384
```

### Configuration File (config/config.yaml)

```yaml
app:
  name: "TradeKnowledge"
  version: "1.0.0"
  debug: false

api:
  auth:
    secret_key: "${JWT_SECRET_KEY}"
    token_expiry_hours: 24
    enable_registration: false
  cors_origins: ["*"]
  rate_limit_per_minute: 60
  enable_docs: true

database:
  sqlite:
    path: "./data/knowledge.db"
  qdrant:
    host: "${QDRANT_HOST:localhost}"
    port: 6333
    collection_name: "tradeknowledge"

embedding:
  model: "nomic-embed-text"
  dimension: 384
  batch_size: 32
```

## 🐳 Docker Deployment

### Development

```bash
# Start development stack
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Access API at http://localhost:8000
```

### Production

```bash
# Set environment variables
export JWT_SECRET_KEY="your-production-secret"
export CORS_ORIGINS="https://yourdomain.com"

# Start production stack
docker-compose up -d

# Initialize API (one-time)
docker-compose exec tradeknowledge-api python scripts/init_api.py --non-interactive
```

## 🏗️ Production Deployment

### Prerequisites

1. **Vector Database**: Qdrant running and accessible
2. **Cache**: Redis for caching (optional but recommended)
3. **Embeddings**: Ollama with nomic-embed-text model
4. **Storage**: Persistent storage for SQLite databases
5. **SSL**: HTTPS certificate for production

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Systemd Service

```ini
[Unit]
Description=TradeKnowledge API
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/tradeknowledge
Environment=PATH=/opt/tradeknowledge/venv/bin
ExecStart=/opt/tradeknowledge/venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📊 Monitoring & Metrics

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Detailed system status (admin only)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/admin/system/status
```

### Prometheus Metrics

The API exposes metrics at `/metrics` for Prometheus monitoring:

- Request latency and count
- Active connections
- Search performance
- Database operation metrics
- Error rates

### Grafana Dashboards

Pre-configured Grafana dashboards are available in `monitoring/grafana/dashboards/`:

- API Performance Dashboard
- User Activity Dashboard
- System Resource Dashboard
- Search Analytics Dashboard

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Change default JWT secret key
- [ ] Disable API documentation in production (`ENABLE_API_DOCS=false`)
- [ ] Configure CORS origins restrictively
- [ ] Set up rate limiting
- [ ] Enable HTTPS only
- [ ] Regular security updates
- [ ] Database backup strategy
- [ ] Log monitoring and alerting

### API Key Management

For service-to-service communication, use API keys:

```python
# Create API key (admin only)
response = requests.post("http://localhost:8000/api/v1/admin/api-keys", 
                        headers={"Authorization": f"Bearer {admin_token}"}, 
                        json={"name": "Service Integration", "scopes": ["read"]})

# Use API key
headers = {"X-API-Key": api_key}
response = requests.post("http://localhost:8000/api/v1/search/query", 
                        headers=headers, json={"query": "trading"})
```

## 🚨 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   # Ensure proper Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/tradeknowledge"
   ```

2. **Database connection errors**
   ```bash
   # Check database permissions
   ls -la data/
   # Reinitialize if needed
   python scripts/init_api.py
   ```

3. **Qdrant connection issues**
   ```bash
   # Check Qdrant status
   curl http://localhost:6333/health
   # Start Qdrant if needed
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Authentication failures**
   ```bash
   # Reset admin password
   python scripts/init_api.py --reset-admin
   ```

### Logs

```bash
# API logs
tail -f logs/api.log

# Access logs
tail -f logs/access.log

# Error logs
tail -f logs/error.log
```

## 🔄 API Versioning

The API uses URL versioning (`/api/v1/`). When breaking changes are needed:

1. Create new version (`/api/v2/`)
2. Maintain backward compatibility for at least 6 months
3. Document migration path
4. Update client libraries

## 📈 Performance Optimization

### Caching Strategy

- **Redis**: Query results, embeddings, user sessions
- **In-memory**: Configuration, frequent searches
- **Database**: FTS5 indexes, query optimization

### Scaling Considerations

- **Horizontal scaling**: Multiple API instances behind load balancer
- **Database scaling**: Read replicas for search queries
- **Vector database**: Qdrant clustering for large datasets
- **CDN**: Static assets and cached responses

## 🎯 Next Steps

1. **Load Testing**: Use Apache Bench or Locust to test under load
2. **Monitoring**: Set up comprehensive monitoring with alerts
3. **Backup Strategy**: Automated database backups
4. **CI/CD Pipeline**: Automated testing and deployment
5. **Client Libraries**: SDKs for popular programming languages

---

**TradeKnowledge Phase 4 API is now production-ready!** 🎉

For issues or questions, check the troubleshooting section or create an issue in the project repository.