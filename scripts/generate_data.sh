for data_dir in `ls /home/jet/Desktop/proj/data/`; do
    echo "Processing directory: $data_dir"
    python scripts/data_converter.py --data_dir /home/jet/Desktop/proj/data/$data_dir
done
echo "All directories processed."