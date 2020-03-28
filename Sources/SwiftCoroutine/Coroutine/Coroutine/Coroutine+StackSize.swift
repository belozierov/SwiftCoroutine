//
//  Coroutine+StackSize.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Darwin

extension Coroutine {
    
    @usableFromInline internal struct StackSize {
        internal let size: Int
    }
    
}

extension Coroutine.StackSize {
    
    internal static let recommended = Coroutine.StackSize(size: Int(SIGSTKSZ))
    internal static let minimal = Coroutine.StackSize(size: Int(MINSIGSTKSZ))
    
    internal static func pages(_ number: Int) -> Coroutine.StackSize {
        .init(size: number * .pageSize)
    }
    
}

