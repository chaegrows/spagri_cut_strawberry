cd ./third_party && \
for dir in *; do \
if [ -d "$dir" ] && [ "$dir" != "livox_ros_driver2" ]; then \
    # mkdir -p "$dir"/build && \
    # echo "Building in $dir..." && \
    # cd "$dir"/build && \
    # cmake .. && \
    # make && \
    # make install && \
    echo "Processing $dir" \
    cd $dir && \
    echo "Current directory: $(pwd)" \
    cd - > /dev/null && \
    echo "Back to previous directory: $(pwd)" \
    # cd /opt/third_party; \
fi \
done
