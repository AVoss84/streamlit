#!/bin/bash
app="claims-dash"
docker build -t ${app} .
docker run -d -p 5000:5000 --name=${app} -v $PWD:/app ${app}   # host-port:container-port