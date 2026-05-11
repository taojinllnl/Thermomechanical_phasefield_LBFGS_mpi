//
//  FileSystem.cpp
//  main
//
//

#include "../../include/Common/FileSystem.h"


using namespace ::dealii;
using namespace ::common;
namespace fs = std::filesystem;



FileSystem::SubDir::SubDir(const std::string& subDir,
                           std::string&      prmDir)
: subDir(subDir)
, prmDir(prmDir)
{}


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

    // 7. if exists already:
    if (fs::exists(normalized)) {
        
        // It must be a directory
        if (fs::is_directory(normalized))
            return true;   // <-------- ALWAYS true if directory exists
        else
            return false;  // exists but is a file → invalid
    }


    // 8. create if not exists
    try {
        if (fs::create_directories(normalized))
            return true; // created successfully
        
        // race condition: directory created between exists() and create_directories()
        return fs::exists(normalized) && fs::is_directory(normalized);
    }
    catch (...) {
        return false;
    }
}



bool FileSystem
::outputDirSystem(const MPIInfo& mpiInfo,
                  std::string& rel_dir,
                  std::string& case_dir_path,
                  std::vector<SubDir>& sub_dirs)
{
    using namespace dealii;
    bool status = false;
    // verify if output dir is existed. if not, create
    status = dir(rel_dir);
    if(!status) return status;
    
    
    // find out the potential subdir name for output
    if (mpiInfo.isRankEqualsTo(0)) {
        case_dir_path  = find_next_numeric_subdir(rel_dir);
    }
    
    if (mpiInfo.isMPI()) {
        case_dir_path = Utilities::MPI::broadcast(*mpiInfo.mpiCommPtr(),
                                                  case_dir_path, 0);
    }
    
    
    // update output dir and sub-dirs
    rel_dir = rel_dir + case_dir_path;
    if (rel_dir.back() != '/') rel_dir.push_back('/');
    
    
    // create sub-dirs
    for (SubDir& subDir : sub_dirs) {
        subDir.prmDir = rel_dir + subDir.subDir;
        if (subDir.prmDir.back() != '/') subDir.prmDir.push_back('/');
        
        if (mpiInfo.isRankEqualsTo(0)) {
            status = dir(subDir.prmDir);
        }
        if(!status) break;
    }
    return status;
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


bool FileSystem::numeric_subdir(std::string& subdir,
                                const std::string& rel_dir)
{
    
    // verify if dir exists
    bool status = dir(rel_dir);
    if (!status) 
    {
        return status;
    }
    
    
    subdir = find_next_numeric_subdir(rel_dir);
    status = dir(rel_dir + subdir);
    
    return status;
}




bool FileSystem::__is_all_digits(const std::string& s)
{
    if (s.empty())
        return false;

    for (unsigned char c : s)
    {
        // must be ASCII number: 0–9
        if (c < '0' || c > '9')
            return false;
    }
    return true;
}




std::string FileSystem::find_next_numeric_subdir(const std::string& path_str)
{
    fs::path path(path_str);
    long long max_val = -1;  // no all digital dir

    try
    {
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (!entry.is_directory())
                continue;

            std::string name = entry.path().filename().string();
            if (!__is_all_digits(name))
                continue;

            long long val = std::stoll(name);  // cast to long long
            if (val > max_val)
                max_val = val;
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return "0";
    }
    catch (const std::out_of_range &)
    {
        // out_of_range of long long
        return "0";
    }


    if (max_val < 0)
        return "0";                // no digital dir

    return std::to_string(max_val + 1);
}
