#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>

#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>

class MyNode {
public:
    MyNode(std::string name) : name(name) {}
    std::string name;
};

void test1() {
    boost::adjacency_list <> g(6);
    MyNode n1("paris");
    MyNode n2("london");
    MyNode n3("new york");

    boost::add_edge(1, 2, g);
    boost::add_edge(1, 3, g);
    boost::add_edge(3, 4, g);
    boost::add_edge(2, 5, g);

    std::ofstream f;
    f.open("/tmp/foo.dot", std::ios_base::out);
    std::map<int, std::string> names;
    for(int i = 0; i < 7; i++) {
        names[i] = "1";
    }
    // std::vector<std::string> names;
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // names.push_back("n1");
    // boost::write_graphviz(f, g, boost::make_label_writer(names));
    boost::write_graphviz(f, g);
    f.close();
    boost::print_graph(g);
}

template <class Name, class Depth>
class my_label_writer {
public:
    my_label_writer(Name _name, Depth _depth) :
        name(_name), depth(_depth) {}
    template <class VertexOrEdge>
    void operator()(std::ostream &os, const VertexOrEdge &v) const {
        os << "[label=\"" << depth[v] << "* @" << name[v] << "\"]";
    }
private:
    Name name;
    Depth depth;
};

template<class Name, class Depth>
my_label_writer<Name, Depth> make_my_label_writer(Name name, Depth depth) {
    return my_label_writer<Name, Depth>(name, depth);
}

struct depth_t {
    typedef boost::vertex_property_tag kind;
};

typedef boost::property<
        boost::vertex_name_t, std::string,
    boost::property<depth_t, int> > VertexProperties;

typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS,
    VertexProperties> Graph;

void mergeChildren(Graph g, int parent) {
    // auto child1 = boost::vertex()
}

void test2() {
    // based on http://fireflyblue.blogspot.co.uk/2008/01/boost-graph-library.html

    Graph g;
    boost::property_map<Graph, boost::vertex_name_t>::type
        name = boost::get(boost::vertex_name, g);
    boost::property_map<Graph, depth_t>::type
        depth = boost::get(depth_t(), g);

    boost::add_edge(1, 2, g);
    boost::add_edge(1, 3, g);
    auto edge = boost::add_edge(1, 4, g);
    boost::add_edge(2, 4, g);

    for(int i = 0; i < 5; i++) {
        std::ostringstream oss;
        oss << i;
        name[boost::vertex(i, g)] = oss.str();
    }

    // name[boost::vertex(2, g)] = "hello";
    depth[boost::vertex(2, g)] = 2;

    depth[boost::vertex(3, g)] = 2;
    depth[boost::vertex(1, g)] = 3;

    // mergeChildren(g, 1);

    // Graph::vertex_iterator vertexIt, vertexEnd;
    // std::tie(vertexIt, vertexEnd) = boost::vertices(g);

    Graph::adjacency_iterator neighborIt, neighborEnd;
    std::tie(neighborIt, neighborEnd) = boost::adjacent_vertices(
        boost::vertex(2, g), g);
    for(; neighborIt != neighborEnd; neighborIt++) {
        std::cout << "child: " << *neighborIt << std::endl;
    }

    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;
    std::cout << "vertices(g) = ";
    for(vp = boost::vertices(g); vp.first != vp.second; vp.first++) {
        std::cout << name[*vp.first] << " ";
    }
    std::cout << std::endl;

    // std::cout << name[v] << " " << depth[v] << std::endl;

    // label_writer<boost::vertex_name_t> mylabelwriter(name);
    auto mylabelwriter = make_my_label_writer(name, depth);

    std::ofstream f("/tmp/foo.dot", std::ios_base::out);
    // boost::write_graphviz(f, g, make_label_writer(name));
    boost::write_graphviz(f, g, mylabelwriter);
    f.close();

    boost::print_graph(g, name);
}

int main(int argc, char *argv[]) {
    // test1();
    test2();

    return 0;
}
