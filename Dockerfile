# Use Python 3.11 base image
FROM python:3.11-slim-bullseye




# Copy both projects
COPY . .
#COPY cv-recommendations-dashboard ./dashboard

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt



# Expose required ports
EXPOSE 3000 5000

# Start supervisor to manage both processes
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]