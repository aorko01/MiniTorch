#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // for std::vector bindings

#include "Tensor.h"
#include "Storage.h"

namespace py = pybind11;

/*
Helper: convert Python list -> std::vector<int>
(pybind can also do this automatically, but this keeps things explicit and predictable)
*/
static std::vector<int> to_vec(const py::object& obj)
{
    std::vector<int> out;
    for (auto item : obj)
    {
        out.push_back(item.cast<int>());
    }
    return out;
}

PYBIND11_MODULE(minitorch_cpp, m)
{
    m.doc() = "MiniTorch C++ backend";

    // -------------------------
    // Storage binding (minimal)
    // -------------------------
    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage")
        .def(py::init<int>())
        .def("data", [](Storage& self) {
            return self.data(); // raw pointer (NOT safe for Python use directly long-term)
        });

    // -------------------------
    // Tensor binding
    // -------------------------
    py::class_<Tensor>(m, "Tensor")

        // constructor: Tensor(shape)
        .def(py::init([](py::object shape) {
            return Tensor(to_vec(shape));
        }))

        // shape()
        .def("shape", &Tensor::shape)

        // strides()
        .def("strides", &Tensor::strides)

        // get(idx)
        .def("get", [](const Tensor& t, py::object idx) {
            return t.get(to_vec(idx));
        })

        // set(idx, value)
        .def("set", [](Tensor& t, py::object idx, float value) {
            t.set(to_vec(idx), value);
        })

        // reshape
        .def("reshape", [](const Tensor& t, py::object new_shape) {
            return t.reshape(to_vec(new_shape));
        })

        // transpose
        .def("transpose", &Tensor::transpose);
}