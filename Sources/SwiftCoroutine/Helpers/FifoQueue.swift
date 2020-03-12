//
//  FifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct FifoQueue<T> {
    
    private var input = [T](), output: [T]
    
    @inlinable init() {
        output = []
    }
    
    @inlinable mutating func push(_ item: T) {
        input.append(item)
    }
    
    mutating func pop() -> T? {
        if let item = output.popLast() { return item }
        switch input.count {
        case 0: return nil
        case 1: return input.popLast()
        default:
            output = input.reversed()
            input.removeAll(keepingCapacity: true)
            return output.popLast()
        }
    }
    
    var isEmpty: Bool {
        input.isEmpty && output.isEmpty
    }
    
}
