//
//  CostIdentifierPool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

class CostIdentifierPool<Cost: AdditiveArithmetic & Comparable & Hashable, Element> {
    
    private class Node {
        let cost: Cost
        public var elements: [Element], next: Node?
        weak var previous: Node?
        
        init(cost: Cost, elements: [Element], next: Node?) {
            self.cost = cost
            self.elements = elements
            self.next = next
        }
        
    }
    
    private let costLimit: Cost
    private var currentCost = Cost.zero
    private var pool = [Cost: Node]()
    private var head, tail: Node?
    
    init(costLimit: Cost) {
        self.costLimit = costLimit
    }
    
    // MARK: - Pool
    
    func pop(_ cost: Cost) -> Element? {
        guard let node = pool[cost] else {
            create(for: cost)
            return nil
        }
        makeHead(node: node)
        if let element = node.elements.popLast() {
            removeTailIfEmpty()
            currentCost -= cost
            return element
        }
        return nil
    }
    
    func push(_ element: Element, for cost: Cost) {
        if cost > costLimit { return }
        currentCost += cost
        freeSpace()
        if let node = pool[cost] {
            node.elements.append(element)
            makeHead(node: node)
        } else {
            create(with: [element], for: cost)
        }
    }
    
    // MARK: - Free
    
    public func removeAll() {
        currentCost = .zero
        pool.removeAll()
        head = nil
        tail = nil
    }
    
    private func freeSpace() {
        while currentCost > costLimit, let node = tail {
            defer { removeTailIfEmpty() }
            while node.elements.popLast() != nil {
                currentCost -= node.cost
                if currentCost <= costLimit { return }
            }
        }
    }
    
    private func removeTailIfEmpty() {
        guard let node = tail, node.elements.isEmpty else { return }
        if node === head { head = nil }
        pool[node.cost] = nil
        node.previous?.next = nil
        tail = node.previous
    }
    
    // MARK: - Nodes
    
    private func makeHead(node: Node) {
        if node === head { return }
        if node === tail { tail = node.previous }
        node.previous?.next = node.next
        node.next?.previous = node.previous
        node.next = head
        head = node
    }
    
    private func create(with elements: [Element] = [], for cost: Cost) {
        let node = Node(cost: cost, elements: elements, next: head)
        head?.previous = node
        head = node
        pool[cost] = node
        if tail == nil { tail = node }
    }
    
}
