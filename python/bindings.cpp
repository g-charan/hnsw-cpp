// Python binding over the header-only index. Thin on purpose: numpy arrays in,
// numpy arrays out, and the GIL dropped while the C++ side searches, so a
// thread pool of Python callers can share one index.
#include "vec/format.hpp"
#include "vec/hnsw.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using Matrix = py::array_t<float, py::array::c_style | py::array::forcecast>;

vec::Metric parse_metric(const std::string& m) {
    if (m == "l2") return vec::Metric::kL2;
    if (m == "ip") return vec::Metric::kInnerProduct;
    throw std::invalid_argument("metric must be 'l2' or 'ip', got '" + m + "'");
}

// Either a freshly built HnswIndex or a MappedIndex; both hand out a GraphView,
// which is all search needs, so the Python class does not care which it holds.
class Index {
public:
    Index(std::size_t dim, std::size_t capacity, const std::string& metric,
          std::size_t M, std::size_t ef_construction)
        : built_(std::make_unique<vec::HnswIndex>(
              dim, capacity, parse_metric(metric),
              vec::HnswIndex::Params{M, ef_construction, 100})) {}

    explicit Index(const std::string& path)
        : mapped_(std::make_unique<vec::MappedIndex>(path)) {}

    void add(Matrix vs) {
        if (!built_) throw std::runtime_error("index was loaded read-only");
        auto b = vs.unchecked<2>();
        if (static_cast<std::size_t>(b.shape(1)) != built_->dim())
            throw std::invalid_argument("vector dim does not match index dim");
        const float* p = vs.data();
        const auto n = static_cast<std::size_t>(b.shape(0));
        py::gil_scoped_release nogil;
        built_->add_many(p, n);
    }

    vec::GraphView view() const { return built_ ? built_->view() : mapped_->view(); }

    py::tuple search_many(Matrix qs, std::size_t k, std::size_t ef) const {
        auto b = qs.unchecked<2>();
        const vec::GraphView g = view();
        if (static_cast<std::size_t>(b.shape(1)) != g.dim)
            throw std::invalid_argument("query dim does not match index dim");
        const auto n = static_cast<std::size_t>(b.shape(0));
        if (k > g.count) k = g.count;

        py::array_t<std::int64_t> ids({n, k});
        py::array_t<float> dists({n, k});
        auto* id_p = ids.mutable_data();
        auto* d_p = dists.mutable_data();
        const float* q_p = qs.data();
        {
            py::gil_scoped_release nogil;
            vec::Scratch scratch;
            for (std::size_t i = 0; i < n; ++i) {
                auto out = vec::search(g, q_p + i * g.dim, k, ef, scratch);
                for (std::size_t j = 0; j < k; ++j) {
                    id_p[i * k + j] = j < out.size() ? out[j].id : -1;
                    d_p[i * k + j] = j < out.size() ? out[j].dist : 0.0f;
                }
            }
        }
        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        if (!built_) throw std::runtime_error("a mapped index is already on disk");
        vec::save_index(*built_, path);
    }

    std::size_t size() const { return view().count; }
    std::size_t dim() const { return view().dim; }

private:
    std::unique_ptr<vec::HnswIndex> built_;
    std::unique_ptr<vec::MappedIndex> mapped_;
};

}  // namespace

PYBIND11_MODULE(hnsw_cpp, m) {
    m.doc() = "HNSW index in C++23 with NEON kernels";
    py::class_<Index>(m, "Index")
        .def(py::init<std::size_t, std::size_t, const std::string&, std::size_t,
                      std::size_t>(),
             py::arg("dim"), py::arg("capacity"), py::arg("metric") = "l2",
             py::arg("M") = 16, py::arg("ef_construction") = 200)
        .def_static("load", [](const std::string& p) { return new Index(p); },
                    py::arg("path"), "Open an index written by save(); mmap-backed, read-only")
        .def("add", &Index::add, py::arg("vectors"))
        .def("search_many", &Index::search_many, py::arg("queries"), py::arg("k") = 10,
             py::arg("ef") = 100)
        .def("search",
             [](const Index& self, Matrix q, std::size_t k, std::size_t ef) {
                 auto shape = q.request().shape;
                 if (shape.size() == 1) q = q.reshape({py::ssize_t{1}, shape[0]});
                 py::tuple r = self.search_many(q, k, ef);
                 return py::make_tuple(r[0].cast<py::array>()[py::int_(0)],
                                       r[1].cast<py::array>()[py::int_(0)]);
             },
             py::arg("query"), py::arg("k") = 10, py::arg("ef") = 100)
        .def("save", &Index::save, py::arg("path"))
        .def("__len__", &Index::size)
        .def_property_readonly("dim", &Index::dim);
}
