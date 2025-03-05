#!/bin/bash

for i in {1..6}
do
  # Define the input and output filenames
  tf_file="prompting/generated_tf/cot_${i}.tf"
  output_file="checkov_output/cot_${i}.txt"

  # Run the checkov command and store the output in the .txt file
  checkov --file "$tf_file" > "$output_file"

  echo "Checkov results saved to $output_file"
done
