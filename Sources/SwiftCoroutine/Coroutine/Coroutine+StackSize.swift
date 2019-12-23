//
//  Coroutine+StackSize.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Darwin

extension Coroutine {
    
    public enum StackSize {
        case recommended
        case minimal
        case pages(Int)
        case bytes(Int)
    }
    
    @inline(__always) public var stackSize: Int {
        context.stackSize
    }
    
}

extension Coroutine.StackSize {
    
    public var size: Int {
        switch self {
        case .recommended: return .recommendedSize
        case .minimal: return .minimalSize
        case .pages(let number): return number * .pageSize
        case .bytes(let bytes): return bytes
        }
    }
    
}

extension Int {
    
    fileprivate static let recommendedSize = Int(SIGSTKSZ)
    fileprivate static let minimalSize = Int(MINSIGSTKSZ)
    
}
