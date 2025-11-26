//
//  FileSystem.cpp
//  main
//
//

#include "FileSystem.h"

namespace fs = std::filesystem;

std::string
FileSystem::pwd()
{
    return fs::current_path().string();
}



bool FileSystem::dir(const std::string& rel_dir)
{
    fs::path base = fs::current_path();
    fs::path p(rel_dir);

    // 1. empty dir is not allowed
    if (rel_dir.empty())
        return false;

    // 2. absolute dir is not allowed
    if (p.is_absolute())
        return false;

    // 3. assemble the target dir path
    fs::path target = base / p;

    // 4. normalize the path, bypass ./、../ etc.
    //    weakly_canonical : return a standardized path if it can be fixed
    fs::path normalized;
    try {
        normalized = fs::weakly_canonical(target);
    } catch (...) {
        return false; // invalid path
    }

    // 5. normalize the pase
    fs::path norm_base = fs::weakly_canonical(base);

    // 6. checke if normalized is under norm_base:
    //    the prefix of normalized should be exactly the same to norm_base
    if (!__is_subpath(norm_base, normalized))
        return false; // invaid, such as starting with "../"

    // 7. if existed：
    if (fs::exists(normalized)) {
        // verify if it is a directory instead of a file
        return fs::is_directory(normalized);
    }

    // 8. create dir, if not exist
    try {
        return fs::create_directories(normalized);
    } catch (...) {
        return false;
    }
}



bool FileSystem::__is_subpath(const std::filesystem::path& base,
                              const std::filesystem::path& child)
{
    namespace fs = std::filesystem;

    auto it_base  = base.begin();
    auto it_child = child.begin();

    for (; it_base != base.end(); ++it_base, ++it_child) {
        if (it_child == child.end())
            return false; // child is shorter than base, impossible to creat a folder under it

        if (*it_base != *it_child)
            return false; // unmatched prefix
    }

    // Completed：
    //  - child == base：current dir is child
    //  - child is under base
    return true;
}
