# Use an official Node.js runtime as the base image
FROM node:16

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the container
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm install

# Copy the rest of the application code to the container
COPY frontend/src ./src
COPY frontend/public ./public

# Build the frontend app
# RUN npm run build

<<<<<<< HEAD
# Expose port 8080 for the frontend
=======
# Expose port 8000 for the frontend
>>>>>>> 8682722311c29b33d7da6f2fe56e26980e39c9f1
EXPOSE 8000

# Start the React app
CMD ["npm", "start"]
