# Install Latest Node Image
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci --omit=dev

# Copy working directory
COPY . .

# Change to non-root user
USER node

# Expose port 3000
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
