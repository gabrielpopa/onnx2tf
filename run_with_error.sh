#!/bin/zsh
RED='\033[0;31m'
GREEN='\033[0;32m'
N='\033[0m' # No Color

# perm_possible=("[0,3,1,2]" "[0,2,1,3]" "[0,3,2,1]")
perm_possible=("[0,3,1,2]" "[0,3,2,1]")

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script runs the convert onnx2tf script."
    echo ""
    echo "Options:"
    echo "  -h                Display this help message and exit."
    echo "  -f FILENAME       Specify the ONNX file to convert."
    echo "  -r FILENAME       Specify the json file for param_replacement_file instructions."
    echo "  -b                Specify batch size 1."
    echo "  -i                Export int8 file."
    echo "  -p 1e-4           PRECISION! Slow, but accurate generation"
    echo "  -s output name    Split model at output name."
    echo ""
    echo "Example:"
    echo "  $0 -f path_to_file.onnx"
}

onnx_file=""
split=""
batch=""
export_int=""
split_at=""
precision=""

while getopts "f:p:r:s:bi" opts; do         
  case "${opts}" in                    # 
    r)         
      repl=${OPTARG}
      ;;
    b)         
      batch="-b 1"
      ;;
    i)         
      export_int="-ei"
      echo "Exporting int8"
      ;;
    s)         
      split_at="${OPTARG}"
      ;;
    f)               
      onnx_file=${OPTARG}
      base_name="${onnx_file%.*}"
      if [ -z "$repl" ]; then
        json_file="${base_name}_generated.json"
      fi  
      ;;
    p)     
      # precision="--disable_strict_mode -cotof -cotoa 1e-4"   
      precision="${OPTARG}"
      ;;
    h)         
      show_help
      exit 1
      ;;
    v)                             
      VERBOSE="verbose=true --print_preinvoke_state=true --print_postinvoke_state=true"     
      ;;
    :)                                    # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      show_help
      exit                       # Exit abnormally.
      ;;
    *)                                    # If unknown (any other) option:
      show_help
      exit                       # Exit abnormally.
      ;;
  esac
done

json_file="${onnx_file%.onnx}_generated.json"
output_file="${onnx_file%.onnx}_output.txt"

# if [ "$#" -eq 1 ]; then
#     json_file="${onnx_file%.onnx}_generated.json"
# elif [ "$#" -eq 2 ]; then
#     onnx_file="$1"
#     json_file="$2"
# elif [ "$#" -gt 2 ]; then
#     onnx_file="$1"
#     json_file="$2"
#     perm_possible=("${@:3}") # Capture all arguments from the third onwards as perm_possible
# else
#     echo "Usage: $0 <onnx_file>"
#     echo "Usage: $0 <onnx_file> [<json_file>]"
#     echo "Usage: $0 <onnx_file> [<json_file>] [<perm_possible>...]"
#     echo ""
#     echo "Example: $0 caformer.onnx caformer.json \"[0,3,1,2]\" \"[0,2,1,3]\" \"[0,3,2,1]\""
#     exit 1
# fi

if [[ "$json_file" != *.json ]]; then
    echo "Error: The json file ($json_file) must have a .json extension."
    exit 2
fi

if [ -z "$onnx_file" ]; then
    show_help
    exit 2
fi

if [[ "$onnx_file" != *.onnx ]]; then
    echo "Error: The onnx file ($onnx_file) must have a .onnx extension."
    exit 2
fi

if [ ! -f "$json_file" ]; then
    echo "{\"operations\": []}" > "$json_file"
    echo "Created $json_file with initial content."
fi

PARMS=("-i" "${onnx_file}" "--param_replacement_file" "${json_file}" "--replace_argmax_to_reducemax_new" "--optimization_for_gpu_delegate" "--not_use_opname_auto_generate" "--disable_group_convolution" "-v debug" "--replace_to_pseudo_operators Erf")

if [ -n "$split_at" ]; then
    echo "Split model at: ${GREEN}$split_at${N}"
    PARMS+=("-onimc" "${split_at}")
fi

if [ -n "$batch" ]; then
    PARMS+=("-b" "1")
fi

if [ -n "$export_int" ]; then
    PARMS+=("-ei")
fi

if [ -n "$precision" ]; then
    PARMS+=("-cotof" "-cotoa" $precision)
else
    PARMS+=("--disable_strict_mode" "-cotof" "-cotoa" "1e-4")
fi

echo "Converting $name: ${GREEN}$onnx_file${N} to tflite with replacement file ${GREEN}$json_file${N}."
echo "PARMS are: ${PARMS[@]}${N}."

command="python3 onnx2tf/onnx2tf.py ${PARMS[@]}"
layer=0
perm_counter=0
current_perm=${perm_possible[$perm_counter]}

# Function to remove ANSI color codes
remove_colors() {
    sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

function check_param_name_exists() {
    local param_name_to_find=$1

    # Use jq to parse the JSON and check for the presence of param_name
    local result=$(jq -r --arg param_name "$param_name_to_find" '
        .operations[] | select(.param_name == $param_name) | .param_name' "$json_file")

    if [[ -n $result ]]; then
        echo "true"
    else
        echo "false"
    fi
}

function add_operation_to_json() {
    local json_file=$1
    local op_name=$2
    local param_name=$3
    local count=$4
    local add_layer=$5

    echo "add_operation_to_json to $add_layer layer"

    local transpose_perm=""
        if [ "$count" == "3" ]; then
            transpose_perm="[0,2,1,3]"
        else
            transpose_perm="[0,2,1]"
        fi

    # Use jq to add the new operation to the operations array
    jq --arg op_name "$op_name" \
       --arg param_name "$param_name" \
       --arg add_layer "$add_layer" \
       --argjson permutation "$transpose_perm" \
       '.operations += [{
           "layer": $add_layer,
           "op_name": $op_name,
           "param_target": "inputs",
           "param_name": $param_name,
           "pre_process_transpose_perm": $permutation
       }]' "$json_file" > tmp.json && mv tmp.json "$json_file"
}

function add_operation_to_json_() {
    local json_file=$1
    local op_name=$2
    local param_name=$3
    local count=$4
    local add_layer=$5

    

    local transpose_perm=""
        if [ "$count" == "3" ]; then
            transpose_perm=${perm_possible[$perm_counter]}
        else
            transpose_perm="[0,2,1]"
        fi

    echo "Add permutation $transpose_perm to $add_layer layer ($param_name)"

    # Use jq to add the new operation to the operations array
    jq --arg op_name "$op_name" \
       --arg param_name "$param_name" \
       --arg add_layer "$add_layer" \
       --argjson permutation "$transpose_perm" \
       '.operations += [{
           "layer": $add_layer,
           "op_name": $op_name,
           "param_target": "inputs",
           "param_name": $param_name,
           "pre_process_transpose_perm": $permutation
       }]' "$json_file" > tmp.json && mv tmp.json "$json_file"
}

function count_elements_in_shape() {
    input_string=$1
    echo "-$input_string-"
    res="${input_string//[^,]}"
    echo "res-$res-"
    echo $res

    echo "${#res}"
}

function delete_operation_from_json() {
    local json_file=$1
    local param_name=$2

    # Use jq to remove the operation with the given param_name
    jq --arg param_name "$param_name" \
       'del(.operations[] | select(.param_name == $param_name))' "$json_file" > tmp.json && mv tmp.json "$json_file"
}

# Function to parse the output and adjust the command or input file
handle_error() {
    local output="$1"
    # output=$(printf "%s" "$output" | remove_colors)
    
    # Example: If the output contains specific errors, handle them accordingly
    if echo "$output" | grep -q "Dimensions must be equal" && echo "$output" | grep -q "Create concrete func"; then
        echo "Dimension mismatch detected AFTER concrete func!"
        echo "Extracting information to update the JSON file..."

        local op_name=$(echo "$output" | grep " onnx_op_name:" | awk '{print $3}')
        if [ "$count" == "" ]; then
            op_name=$(echo "$output" | grep " name=")
            echo $op_name
            op_name=$(echo "$output" | sed -n "s/.*name='\(.*\)'.*/\1/p")
        fi
        echo "Looking to fix: ${op_name}"

        # Works :|
        # local op_string=$(echo "$output" | sed -n "\#onnx_op_type.*${op_name}#,+6p")

        # Works :)
        # local op_string=$(echo "$output" | sed -n "\#onnx_op_name.*${op_name}#,/\(onnx_op_name\)/{/\(onnx_op_name\)/!p;}")
        local op_string=$(echo "$output" | remove_colors | sed -n "\|onnx_op_name: ${op_name}$|,/\(onnx_op_name\)/{/\(onnx_op_name\)/!p;}")

        if [ $? -ne 0 ]; then
          echo "Error: search for $op_name was not successful."
          return 2
        fi
        local last_line=$(echo "$op_string" | tail -n 1)
        echo "op_string: $op_string"
        echo "first_line: $first_line"
        echo "last_line: $last_line"
        first_number=$(echo "$last_line" | remove_colors | awk '{print $2}')
        echo "first_number: $first_number"
        if [[ "$first_number" =~ ^[0-9]+$ ]]; then
            problem_layer=$((first_number - 1))
        else
            problem_layer=$problem_layer
        fi
        echo "problem_layer: $problem_layer"

        if [[ problem_layer -gt layer ]]; then
            echo "ADVANCED from $layer to: $problem_layer"
            perm_counter=0
        fi
        layer=$problem_layer

        # Initialize arrays
        input_names=()
        input_shapes=()

        # Temporary variables to hold names and shapes
        current_name=""
        current_shape=""

        # Extract the names and shapes
        while IFS= read -r line; do
            clean_line=$(echo "$line" | remove_colors)

            # Match the input_name line to extract the name
            if [[ "$clean_line" =~ input_name\.[0-9]+:\ ([^ ]+)\ shape:\ ([^ ]+) ]]; then
                current_name="${BASH_REMATCH[1]}"
                # echo "found name:$current_name"
                input_names+=("$current_name")
            fi

            shape=$(echo "$clean_line" | sed -n 's/.*shape: (\([^)]*\)).*/\1/p')


            if [[ -n "$shape" ]]; then
                input_shapes+=("${shape}")
            fi

            # Match the input.x line to extract the shape
            # if [[ "$clean_line" =~ input\.[0-9]+\.[xy]:\ name:\ [^ ]+\ shape:\ \(([^)]+)\) ]]; then
            #     current_shape="(${match[1]})"
            #     input_shapes+=("$current_shape")
            # fi

            # if [[ "$clean_line" =~ input_name\.[0-9]+:\ ([^ ]+)\ shape:\ (\[[^]]+\]) ]]; then
            #     echo ${BASH_REMATCH[0]}
            #     echo ${BASH_REMATCH[1]}
            #     echo ${BASH_REMATCH[2]}
            #     exit 1
            #     input_names+=("${BASH_REMATCH[1]}")
            #     input_shapes+=("${BASH_REMATCH[2]}")
            # fi
        done <<< "$op_string"

        # Print arrays to verify
        # echo "Input Names: ${input_names[@]}"
        # echo "Input Shapes: ${input_shapes[@]}"

        found=false
        deleted=false

        for ((i=0; i<${#input_names[@]}; i++)); do
            
            input_name=${input_names[$i]}
            input_shape=${input_shapes[$i]}
            # echo "Name: ${input_name}, Shape: ${input_shape}"
            if [[ $(check_param_name_exists $input_name) == "false" ]]; then
                # echo "input_shape: $input_shape"
                # count_elements_in_shape $input_shape
                local count=0
                # if [ ${#input_shape} == "(4,)" ]; then
                #     echo "found (4,)"
                #     count="3"
                # elif [ ${#input_shape} == "4," ]; then
                #     echo "found 4,"
                #     count="3"
                # else
                #     echo "seach for commas"
                #     count="${input_shape//[^,]}"
                #     count="${#count}"
                #     echo "found $count commas"
                # fi

                count="${input_shape//[^,]}"
                count="${#count}"
                
                # echo "shape count:${count}"
                if [ ${count} == "0" ]; then
                    echo "skip 1 size shape"
                elif [ ${count} == "1" ]; then
                    echo "skip 2 size shape"
                else
                    found=true
                    add_operation_to_json_ $json_file $op_name $input_name ${count} $layer
                    echo "JSON file updated with new operation: $op_name input: $input_name shape: ${count}"
                    return 0
                fi
            else
                deleted=true
                delete_operation_from_json $json_file $input_name
                echo "DELETED: $input_name"
            fi
        done

        if [[ "$deleted" == true && "$found" == false ]]; then
            ((perm_counter++))
            if [[ $perm_counter -ge ${#perm_possible[@]} ]]; then
                echo "Tried all permutations. :( (${perm_possible[@]})"
                exit 4
                perm_counter=0
            fi
            echo "DELETED ALL PERMUTATIONS. Now using: ${perm_possible[$perm_counter]} ($perm_counter)"
        fi
        

    elif echo "$output" | grep -q "Dimensions must be equal"; then
        echo "Dimension mismatch detected BEFORE concrete func!"
        echo "Extracting information to update the JSON file..."
        # Extract the necessary information from the output

        local op_name=$(echo "$output" | grep " onnx_op_name:" | awk '{print $3}')
        if [ "$count" == "" ]; then
            op_name=$(echo "$output" | grep " name=")
            echo $op_name
            op_name=$(echo "$output" | sed -n "s/.*name='\(.*\)'.*/\1/p")
        fi
        echo "Looking to fix: ${op_name}"

        # local op_string=$(echo "$output" | sed -n "\#onnx_op_type.*${op_name}#, +6p")
        # Step 1: Find the line number where the pattern matches
        match_line=$(echo "$output" | remove_colors | grep -n "onnx_op_type.*${op_name}" | head -n1 | cut -d: -f1)

        # Check if a match was found
        if [[ -z "$match_line" ]]; then
          echo "No match found for op_name: ${op_name}"
          exit 1
        fi

        # Step 2: Calculate the starting line (two lines before the match)
        start_line=$((match_line - 2))

        # Ensure the starting line is at least 1
        if (( start_line < 1 )); then
          start_line=1
        fi

        local op_string=$(echo "$output" | remove_colors | sed -n "\#onnx_op_type.*${op_name}#,+9p")
        if [ $? -ne 0 ]; then
          echo "Error: search for $op_name was not successful."
          return 2
        fi

        first_number=$(echo "$output" | remove_colors | sed -n "${start_line},+1p" | awk '{print $2}' | tr -d '\n')
        problem_layer=0
        if [[ "$first_number" =~ ^[0-9]+$ ]]; then
            problem_layer=$((first_number))
        else
            problem_layer=$problem_layer
        fi
        echo "problem_layer: $problem_layer"

        if [[ problem_layer -gt layer ]]; then
            echo "ADVANCED from $layer to: $problem_layer"
            perm_counter=0
        fi
        layer=$problem_layer

        # Initialize arrays
        input_names=()
        input_shapes=()

        # Temporary variables to hold names and shapes
        current_name=""
        current_shape=""

        # Extract the names and shapes
        while IFS= read -r line; do
            clean_line=$(echo "$line" | remove_colors)
            if [[ "$clean_line" =~ input_name\.[0-9]+:\ ([^ ]+)\ shape:\ (\[[^]]+\]) ]]; then
                input_names+=("${BASH_REMATCH[1]}")
                input_shapes+=("${BASH_REMATCH[2]}")
            fi
        done <<< "$op_string"

        # Print arrays to verify
        # echo "Input Names: ${input_names[@]}"
        # echo "Input Shapes: ${input_shapes[@]}"

        found=false
        deleted=false

        for ((i=0; i<${#input_names[@]}; i++)); do
            
            input_name=${input_names[$i]}
            input_shape=${input_shapes[$i]}
            # echo "Name: ${input_name}, Shape: ${input_shape}"
            if [[ $(check_param_name_exists $input_name) == "false" ]]; then
                # echo "input_shape: $input_shape"
                # count_elements_in_shape $input_shape
                local count=0

                count="${input_shape//[^,]}"
                count="${#count}"

                if [[ "$count" =~ ^[0-9]+$ ]]; then
                    commas=$((count))
                else
                    echo "Count for shape not a number: ${count} for $op_name input: $input_name"
                    return 5
                fi
                
                # echo "shape count:${count}"
                if [ ${count} == "0" ]; then
                    echo "skip 1 size shape"
                elif [ ${count} == "1" ]; then
                    echo "skip 2 size shape"
                else
                    if [[ commas -gt 3 ]]; then
                        echo "skip ${count} shape size"
                        return 6
                    else
                        found=true
                        add_operation_to_json_ $json_file $op_name $input_name ${count} $layer
                        echo "JSON file updated with new operation: $op_name input: $input_name shape: ${count}"
                        return 0
                    fi
                fi
            else
                deleted=true
                delete_operation_from_json $json_file $input_name
                echo "DELETED: $input_name"
            fi
        done

        if [[ "$deleted" == true && "$found" == false ]]; then
            ((perm_counter++))
            if [[ $perm_counter -ge ${#perm_possible[@]} ]]; then
                echo "Tried all permutations. :( (${perm_possible[@]})"
                exit 4
                perm_counter=0
            fi
            echo "DELETED ALL PERMUTATIONS. Now using: ${perm_possible[$perm_counter]} ($perm_counter)"
        elif [[ "$deleted" == false && "$found" == false ]]; then
            echo "ERROR: Could not find solution. Unsupported shapes."
            exit 7
        fi
        
    elif echo "$output" | grep -q "File specified in param_replacement_file not found."; then
        echo "File specified in param_replacement_file not found. "
        touch $json_file
        return 1
    elif echo "$output" | grep -q "The file specified in param_replacement_file is not in JSON format"; then
        echo "{\"operations\": []}" > $json_file
    else
        echo "ERROR: Unhandled error. Exiting..."
        return 1
    fi
    
    return 0
}

while true; do
    echo "Running conversion..."
    output=$(eval "$command" 2>&1 | tee /dev/tty $output_file)
    statuss="0"
    if [ -n "$ZSH_VERSION" ]; then
        statuss=${pipestatus[1]}
    else
        statuss=${PIPESTATUS[0]}
    fi

    echo "Status $statuss"
    # if [ $statuss -eq 0 ]; then
    if echo "$output" | grep -q "Float16 tflite output complete"; then
        echo "Done!"
        break
    else
        echo "Conversion failed. Handling error..."
        handle_error "$output"
        if [ $? -ne 0 ]; then
            echo "Exiting due to unhandled error."
            exit 3
        fi
    fi

    echo "Retrying conversion with new permutation parameters..."
done
