FROM maven:3.9.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY identity-service/pom.xml ./pom.xml
COPY identity-service/src ./src
RUN mvn -f ./pom.xml clean package -DskipTests

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
EXPOSE 8080
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /otel && \
		curl -L -o /otel/opentelemetry-javaagent.jar \
			https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v1.28.0/opentelemetry-javaagent.jar
ENTRYPOINT ["java", "-javaagent:/otel/opentelemetry-javaagent.jar", "-jar", "app.jar"]