FROM eclipse-temurin:17-jre
WORKDIR /work

# Use published Alloy dist jar (6.2.0)
ADD https://github.com/AlloyTools/org.alloytools.alloy/releases/download/v6.2.0/org.alloytools.alloy.dist.jar /opt/alloy/alloy.jar

ENTRYPOINT ["java", "-jar", "/opt/alloy/alloy.jar"]
