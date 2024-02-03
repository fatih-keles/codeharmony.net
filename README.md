# [codeharmony.net](https://codeharmony.net)
This is a `Flask` web application that is deployed on `Container Instances` on `OCI`.

## Initialize and commit
```bash
echo "# codeharmony.net" >> README.md
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/fatih-keles/codeharmony.net.git
git push -u origin main
```

## Commit changes
```bash
git add .
git commit -m "removed pip freeze file"
git push -u origin main
```

## Clone and run
```bash
git clone https://github.com/fatih-keles/codeharmony.net.git
flask --app run run --debug
```

## Build the container and run
- Prepare your .env file
```bash
BUILD_TAG=lhr.ocir.io/<namespace>/flask-apps/codeharmony:$BUILD_NUMBER
USER="<username>"
PASSWORD="<auth-token>"
## CI_DEMO
COMPARTMENT_ID=ocid1.compartment.oc1...
REPOSITORY_NAME=flask-apps/codeharmony
## CI_DEMO_VCN
VCN_OCID=ocid1.vcn.oc1.uk-london-1..
SUBNET_ID=ocid1.subnet.oc1.uk-london-1..
NSG_ID=ocid1.networksecuritygroup.oc1.uk-london-1..
AD_NAME=sbnF:UK-LONDON-1-AD-1
## CI_DEMO_INSTANCE
CI_INSTANCE_NAME=CI_CODEHARMONYNET:$BUILD_NUMBER
CI_SHAPE=CI.Standard.E4.Flex
CI_SHAPE_CONFIG='{"ocpus": 1,"memoryInGBs": 8}'
CONTAINERS='[{"imageUrl": "'$BUILD_TAG'", "command": [""], "arguments": ["", ""]}]'
```
- run build file  
```bash
./build-and-deploy.sh
```
- use commented scripts in the build file
```bash
# docker run --rm -it -p 5000:5000 $BUILD_TAG 
# docker run --rm -d -p 5000:5000 $BUILD_TAG 
```