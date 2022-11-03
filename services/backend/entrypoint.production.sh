#!/bin/sh

echo "Downloading embeddings file"
OUTPUTFILE=/usr/src/app/seed_data/data/prompt_to_embedding.64.100.1000.pkl
check_embedding_hash () {
	myhash=$(sha256sum $OUTPUTFILE | cut -d' ' -f1)
	if [ $myhash = "b86dc682bd181725129b3223569b10ba3375d089a77b02721ab0b717c29daa1c" ]; then
		return 0; # 0 = true
	fi
	return 1; # 1 = false
}
download_embedding_file () {
  	wget \
	  --no-verbose --show-progress \
	  --progress=bar:force:noscroll \
	  -O $OUTPUTFILE \
	  https://github.com/kennethgoodman/Columbia-E4579/raw/main/services/backend/seed_data/data/prompt_to_embedding.64.100.1000.pkl
	if check_embedding_hash; then
		echo "checksum correct"
	else
		echo "checksum incorrect, deleting file, try again, if this persists, ask for help"
		rm $OUTPUTFILE
		exit 1
	fi
}
if test -f "$OUTPUTFILE"; then
	if check_embedding_hash; then
		echo "$OUTPUTFILE exists, not downloading"
	else
		echo "$OUTPUTFILE exists, but wrong checksum, redownloading"
		rm $OUTPUTFILE
		download_embedding_file
	fi
else
	download_embedding_file
fi

gunicorn -b 0.0.0.0:5000 wsgi:app --access-logfile -
