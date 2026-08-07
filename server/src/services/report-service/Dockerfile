# Install Java and Maven for the Spring Boot service
FROM maven:3.9.9-eclipse-temurin-17 AS builder

# Set the working directory
WORKDIR /app

# Copy the Maven build file and install dependencies
COPY pom.xml ./

# Download dependencies to speed up the build process
RUN mvn dependency:go-offline

# Copy the rest of the application code
COPY src ./src

# Build the application
RUN mvn clean package


FROM eclipse-temurin:17-jre

WORKDIR /app

COPY --from=builder /app/target/*.jar app.jar

# Expose the application port
EXPOSE 8081

# Start the application
ENTRYPOINT ["java", "-jar", "app.jar"]