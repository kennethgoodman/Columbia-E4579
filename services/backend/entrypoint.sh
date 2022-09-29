#!/bin/sh

# Initialise option flag with a false value
OPT_RECREATE=0

# Process all options supplied on the command line
while getopts ':r' 'OPTKEY'; do
    case ${OPTKEY} in
        'r')
            # Update the value of the option x flag we defined above
            OPT_RECREATE=1
            ;;
    esac
done

echo "Downloading embeddings file"
OUTPUTFILE=/usr/src/app/seed_data/data/prompt_to_embedding.64.100.1000.pkl
if test -f "$OUTPUTFILE"; then
	echo "$OUTPUTFILE exists, not downloading"
else
	wget \
	  --no-verbose --show-progress \
	  --progress=bar:force:noscroll \
	  -O $OUTPUTFILE \
	  https://github.com/kennethgoodman/Columbia-E4579/raw/main/services/backend/seed_data/data/prompt_to_embedding.64.100.1000.pkl
fi

echo "Waiting for mysql..."

while ! nc -z api-db 3306; do
  sleep 0.1
done

echo "mysql started"

if [ "$OPT_RECREATE" -eq 1 ]; then
	echo "recreating db"
	python manage.py recreate_db
	echo "db created"
	python manage.py seed_db
	echo "db seeded"
else
	echo "not recreating because didn't pass -r"
fi

python manage.py run -h 0.0.0.0
