# 1. Copy the template notebook for experiments and rename it and the first heading to the given name
read -p "Enter the experiment name: " name
cp nbs/experiments/_template.ipynb "nbs/experiments/${name}".ipynb 

# Replace hyphens with spaces
formattedString=$(echo "$name" | sed 's/-/ /g')

# Capitalize each word
outputString=$(echo "$formattedString" | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2));}1')

sed -i "s/# Experiment Name/# ${outputString}/g" "nbs/experiments/${name}".ipynb 

# 2. Add the notebook to the jupyter cache index
jcache notebook add "nbs/experiments/${name}".ipynb 