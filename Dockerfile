# Use Python 3.11 base image
FROM python:3.11-slim-bullseye

# Install Node.js 18 and required system dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install supervisor for process management
RUN apt-get update && \
    apt-get install -y supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy both projects
COPY cv-recommendations-server ./server
COPY cv-recommendations-dashboard ./dashboard

# Install Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Install Node.js dependencies and build Next.js
RUN cd dashboard && \
    npm install && \
    npm run build

# Configure supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose required ports
EXPOSE 3000 5000

# Start supervisor to manage both processes
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]