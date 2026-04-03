#include <limits.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    uint32_t size = PATH_MAX;
    char executable_path[PATH_MAX];
    char resolved_path[PATH_MAX];
    char dir_path[PATH_MAX];
    char launch_script[PATH_MAX];

    if (_NSGetExecutablePath(executable_path, &size) != 0) {
        fprintf(stderr, "Failed to get executable path.\n");
        return 1;
    }

    if (realpath(executable_path, resolved_path) == NULL) {
        perror("realpath");
        return 1;
    }

    if (strlen(resolved_path) >= sizeof(dir_path)) {
        fprintf(stderr, "Executable path is too long.\n");
        return 1;
    }

    strcpy(dir_path, resolved_path);

    if (snprintf(launch_script, sizeof(launch_script), "%s/../Resources/launch.sh", dirname(dir_path)) >= (int)sizeof(launch_script)) {
        fprintf(stderr, "Launch script path is too long.\n");
        return 1;
    }

    execl("/bin/bash", "bash", launch_script, (char *)NULL);
    perror("execl");
    return 1;
}
