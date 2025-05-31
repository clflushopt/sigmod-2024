CMAKE_BUILD_DIR=build
mkdir -p $CMAKE_BUILD_DIR

cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "Build successful, running tests..."
./build/test