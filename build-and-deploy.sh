BUILD_NUMBER=5
echo "Building and deploying version $BUILD_NUMBER"

# import the environment variables
echo "Importing environment variables"
source .env

## Build the image 
echo "Building the image"
docker build --tag $BUILD_TAG --file Dockerfile .

## List images 
echo "Listing images"
docker images 

# docker run -e OAK=$OPENAI_API_KEY --rm -it -p 5001:5001 $BUILD_TAG bash
# exit 0

## Run container locally 
# docker run --rm -it -p 5000:5000 $BUILD_TAG 
# docker run -e OAK=$OPENAI_API_KEY --rm -it -p 5000:5000 $BUILD_TAG 
# docker run --rm -d -p 5000:5000 $BUILD_TAG 
# curl -s -w "%{size_download}\n" -o /dev/null http://localhost:5000
# exit 0

## Make sure repository exists
echo "Creating repository"
oci artifacts container repository create --compartment-id $COMPARTMENT_ID --display-name $REPOSITORY_NAME

## Login to repository
echo "Logging in to repository"
docker login lhr.ocir.io -u $USER -p $PASSWORD
## docker tag split-pdf:1 lhr.ocir.io/lrfymfp24jnl/flask-apps/split-pdf:1

## Push the container 
echo "Pushing the container"
docker push $BUILD_TAG

# echo "'$VNICS'"
# echo $AD_NAME 
# echo $CI_SHAPE
# echo "'$CI_SHAPE_CONFIG'"
# echo "'$CONTAINERS'"

# oci container-instances container-instance create --debug --compartment-id $COMPARTMENT_ID --availability-domain $AD_NAME --shape $CI_SHAPE --shape-config "'$CI_SHAPE_CONFIG'" --containers "'$CONTAINERS'" --vnics "'$VNICS'"


## Create the container instance
echo "Creating the container instance"
# oci container-instances container-instance create --display-name $CI_INSTANCE_NAME --compartment-id $COMPARTMENT_ID --availability-domain $AD_NAME --shape $CI_SHAPE --shape-config '{"ocpus": 1,"memoryInGBs": 1}' --vnics '[{"subnetId": "'"$SUBNET_ID"'", "nsg_ids" : ["'"$NSG_ID"'"]}]' --containers '[{"imageUrl": "'"$BUILD_TAG"'" }]'

containers_json='[{"imageUrl": "'"$BUILD_TAG"'", "displayName": "'"$CI_INSTANCE_NAME"'", "environment-variables": {"OAK": "'"$OPENAI_API_KEY"'", "DD_API_KEY": "'"$DD_API_KEY"'","DD_SITE": "'"$DD_SITE"'", } }]'
shape_config='{"ocpus": 2,"memoryInGBs": 16}'
vnics_json='[{"subnetId": "'"$SUBNET_ID"'", "nsg_ids" : ["'"$NSG_ID"'"]}]'

oci container-instances container-instance create --display-name $CI_INSTANCE_NAME --compartment-id $COMPARTMENT_ID --availability-domain $AD_NAME --shape $CI_SHAPE --shape-config "$shape_config" --vnics "$vnics_json" --containers "$containers_json"
