#pragma once
#include <iostream>
#include <unordered_map>
#include <memory>
#include <array>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>
#include <functional>

namespace Solver {

    template <typename T>
    class Node {
    public:
        Node(T data) : data(data) {
            connections.clear();
        }

        Node(T data, std::shared_ptr<Node> node, double dist) : data(data) {
            connections[node] = dist;
        }

        double get_distance_to(const std::shared_ptr<Node>& N) const {
            auto it = connections.find(N);
            if (it != connections.end()) {
                return it->second;
            }
            return std::numeric_limits<double>::infinity();
        }

        void add_connection_to(std::shared_ptr<Node> N, double dist) {
            connections[N] = dist;
        }

        T get_data() const { return data; }

        bool operator==(const Node<T>& other) const {
            return data == other.data;
        }

        struct Hash {
            std::size_t operator()(const std::shared_ptr<Node<T>>& node) const {
                return std::hash<T>()(node->get_data());
            }
        };

    private:
        T data;
        std::unordered_map<std::shared_ptr<Node<T>>, double, Node<T>::Hash> connections;
    };

    template<typename T, size_t N>
    struct Map {
        Map(const std::unordered_set<std::shared_ptr<Node<T>>, typename Node<T>::Hash>& nodes,
            const std::array<std::shared_ptr<Node<T>>, N>& pet_nodes,
            const std::array<std::shared_ptr<Node<T>>, N>& house_nodes)
            : Nodes(nodes), pet_nodes(pet_nodes), house_nodes(house_nodes) {
            for (const auto& node : Nodes) {
                data_to_node[node->get_data()] = node;
            }
        }

        std::shared_ptr<Node<T>> get_by_data(const T& data) const {
            auto it = data_to_node.find(data);
            if (it != data_to_node.end()) {
                return it->second;
            }
            throw std::runtime_error("Node not found");
        }

        std::array<std::shared_ptr<Node<T>>, N> pet_nodes;
        std::array<std::shared_ptr<Node<T>>, N> house_nodes;
        std::unordered_set<std::shared_ptr<Node<T>>, typename Node<T>::Hash> Nodes;
        std::unordered_map<T, std::shared_ptr<Node<T>>> data_to_node;
        std::shared_ptr<Node<T>> car_node;
    };

    template<typename T, size_t N>
    inline bool is_route_valid(const Map<T, N>& the_map, const std::vector<std::shared_ptr<Node<T>>>& route, int capacity = 4) {
        int cargo = 0;
        std::unordered_set<T> collected_pets;

        for (const auto& node : route) {
            T data = node->get_data();
            if (std::find_if(the_map.pet_nodes.begin(), the_map.pet_nodes.end(),
                [&data](const std::shared_ptr<Node<T>>& pet) { return pet->get_data() == data; }) != the_map.pet_nodes.end()) {
                collected_pets.insert(data);
                cargo++;
            }
            if (std::find_if(the_map.house_nodes.begin(), the_map.house_nodes.end(),
                [&data](const std::shared_ptr<Node<T>>& house) { return house->get_data() == data; }) != the_map.house_nodes.end() &&
                collected_pets.find(tolower(data)) != collected_pets.end()) {
                cargo--;
            }
            if (cargo > capacity) {
                return false;
            }
            if (isupper(data) && collected_pets.find(tolower(data)) == collected_pets.end()) {
                return false; // House visited before collecting the corresponding pet
            }
        }
        return true;
    }

    template<typename T, size_t N>
    double get_distance_for_route(const std::vector<std::shared_ptr<Node<T>>>& route) {
        double d = 0;
        for (size_t i = 1; i < route.size(); ++i) {
            d += route[i - 1]->get_distance_to(route[i]);
        }
        return d;
    }

    template<typename T, size_t N>
    std::pair<double, std::vector<std::shared_ptr<Node<T>>>> find_shortest_route_for_capacitated_car(const Map<T, N>& the_map, int car_capacity) {
        std::vector<std::shared_ptr<Node<T>>> nodes;
        nodes.reserve(the_map.Nodes.size());
        for (const auto& node : the_map.Nodes) {
            nodes.push_back(node);
        }

        double min_distance = std::numeric_limits<double>::infinity();
        std::vector<std::shared_ptr<Node<T>>> best_route;

        std::sort(nodes.begin() + 1, nodes.end(), [](const std::shared_ptr<Node<T>>& a, const std::shared_ptr<Node<T>>& b) { return a->get_data() < b->get_data(); });
        do {
            if (islower(nodes[0]->get_data())) continue; // Skip routes that start with a house

            std::vector<std::shared_ptr<Node<T>>> route = { the_map.car_node };
            route.insert(route.end(), nodes.begin(), nodes.end());

            if (is_route_valid(the_map, route, car_capacity)) {
                double distance = get_distance_for_route(route);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_route = route;
                }
            }
        } while (std::next_permutation(nodes.begin() + 1, nodes.end(), [](const std::shared_ptr<Node<T>>& a, const std::shared_ptr<Node<T>>& b) { return a->get_data() < b->get_data(); }));

        return { min_distance, best_route };
    }
}
