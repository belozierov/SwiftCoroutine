//
//  LifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

//internal struct LifoQueue<T> {
//
//    private typealias Pointer = UnsafeMutablePointer<Node>
//
//    private struct Node: QueueNode {
//        var item: T?
//        var next = 0
//        var nextToFree = 0
//    }
//
//    private var head = 0
//    private var eraser = QueueNodeEraser<Node>()
//
//    // MARK: - Push
//
//    internal mutating func push(_ item: T) {
//        let node = Pointer.allocate(capacity: 1)
//        node.initialize(to: Node(item: item))
//        eraser.startAccess()
//        atomicUpdate(&head) {
//            node.pointee.next = $0
//            return Int(bitPattern: node)
//        }
//        eraser.endAccess()
//    }
//
//    // MARK: - Pop
//
//    internal mutating func pop() -> T? {
//        if head == 0 { return nil }
//        var node: Pointer?
//        eraser.startAccess()
//        defer { eraser.endAccess(node) }
//        atomicUpdate(&head) {
//            node = Pointer(bitPattern: $0)
//            return node?.pointee.next ?? 0
//        }
//        defer { node?.pointee.item = nil }
//        return node?.pointee.item
//    }
//
//    // MARK: - Free
//
//    internal mutating func free() {
//        var address = head
//        while let node = Pointer(bitPattern: address) {
//            address = node.pointee.next
//            node.deinitialize(count: 1).deallocate()
//        }
//        eraser.free()
//    }
//
//}
