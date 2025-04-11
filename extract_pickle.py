import pickle
import argparse
import os

def read_prompts(file_path):
    """Load neural exec prompts from file, supporting both pickle and text formats"""
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

def pickle_to_txt(input_path, output_path, verbose=False):
    """Extract prompts from pickle file and save to text file"""
    try:
        # Load prompts from pickle
        prompts = read_prompts(input_path)

        if verbose:
            print(f"Loaded {len(prompts)} prompts from {input_path}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write prompts to text file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, prompt in enumerate(prompts):
                outfile.write(f"{prompt}\n")

                if verbose and i < 3:
                    print(f"Sample prompt {i+1}: {prompt[:100]}...")

        print(f"Successfully extracted {len(prompts)} prompts to {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Extract prompts from pickle file to text')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pickle file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output text file (default: input_file_name.txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print additional information during processing')

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}.txt"

    # Process the file
    success = pickle_to_txt(args.input, args.output, args.verbose)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())