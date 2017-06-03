cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    copts = ['-Ilib/inc'],
    deps = [
        ":hello-time",
        "//lib:hello-greet",
    ],
)
