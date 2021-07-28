# Change ROOTPATH to cwd
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
SHELL_FOLDER=${SHELL_FOLDER//\//\\/}
COMMAND=$"s/ROOTPATH/${SHELL_FOLDER}/g"
sed -i $COMMAND ./root_config.json
echo Done!