//
//  QueueNodeEraser.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal protocol QueueNode {
    
    var nextToFree: Int { get set }
    
}

internal struct QueueNodeEraser<T: QueueNode> {
    
    private var accessCount = 0
    private var toFree = 0
    
    @inlinable internal mutating func startAccess() {
        atomicAdd(&accessCount, value: 1)
    }
    
    internal mutating func endAccess(_ node: UnsafeMutablePointer<T>? = nil) {
        let freeFrom = toFree
        if atomicAdd(&accessCount, value: -1) == 1 {
            if freeFrom != 0, atomicCAS(&toFree, expected: freeFrom, desired: 0) {
                freeNodes(freeFrom)
            }
            node?.deinitialize(count: 1).deallocate()
        } else if let node = node {
            atomicUpdate(&toFree) {
                node.pointee.nextToFree = $0
                return Int(bitPattern: node)
            }
        }
    }
    
    @inlinable internal func free() {
        if toFree != 0 { freeNodes(toFree) }
    }
    
    private func freeNodes(_ address: Int) {
        var address = address
        while let node = UnsafeMutablePointer<T>(bitPattern: address) {
            address = node.pointee.nextToFree
            node.deinitialize(count: 1).deallocate()
        }
    }
    
}
