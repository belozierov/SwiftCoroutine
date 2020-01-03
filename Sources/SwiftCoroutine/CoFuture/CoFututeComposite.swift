//
//  CoFututeComposite.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

@_functionBuilder
public struct CoFututeComposite<T> {
    
    public static func buildBlock(_ components: CoFuture<T>...) -> [CoFuture<T>] {
        components
    }
    
}
