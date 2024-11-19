#!/bin/bash

selectRegion() {
	PS3="Select region: "
	regions=("us-east-1" "us-east-2" "us-west-1" "us-west-2" "eu-north-1")
	select region in "${regions[@]}"; do
		case "${region}" in
			"us-east-1")
				printf "${region}\n"
				break
				;;
			"us-east-2")
				printf "${region}\n"
				break
				;;
			"us-west-1")
				printf "${region}\n"
				break
				;;
			"us-west-2")
				printf "${region}\n"
				break
				;;
			"eu-north-1")
				printf "${region}\n"
				break
				;;
		esac
	done
}

if ! which aws 2> /dev/null; then
	printf "aws cli not installed\n"
else
	reg="$(selectRegion)"
	aws s3 sync --no-sign-request --region "${region}" "s3://cse-cic-ids2018/" .
fi
